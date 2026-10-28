import time
from nkg.models.Graph import Graph
from nkg.models.index_objects import *
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed
import dspy
from nkg.index.extraction.extract_relations import *
from nkg.utils.general import batch_list
from nkg.utils.math_utils import sample_results
from nkg.index.construction.checkpointing import ConstructionCheckpoint
from nkg.utils.token_tracker import usage_tokens


def _bulk_fact_edge_constructor(kind: str, use_cot: bool, neutral_description: bool, edge_descriptions: bool = True,
                                fact_edge_prompt: str = "default"):
    """DSPy module for bulk fact->fact edges. kind: "intra" (same chunk) or "global" (clustered cross-chunk).
    edge_descriptions=False -> the label-only signatures (no description sentence is generated).
    fact_edge_prompt="event" -> BulkIntraChunkFactEdgesEvent (also given the chunk's text) / BulkGlobalFactEdgesEvent
    (extract_relations.EVENT_FACT_EDGE_GUIDANCE: RED / Hong et al. relation families + WILDCARD, specific labels,
    few-shot), label-only."""
    if fact_edge_prompt == "event":
        if edge_descriptions:
            raise ValueError("fact_edge_prompt='event' is label-only: use it with edge_descriptions=False")
        sig = BulkIntraChunkFactEdgesEvent if kind == "intra" else BulkGlobalFactEdgesEvent
    elif fact_edge_prompt != "default":
        raise ValueError(f"fact_edge_prompt must be 'default' or 'event', got {fact_edge_prompt!r}")
    elif not edge_descriptions:
        sig = BulkIntraChunkFactEdgesLabelOnly if kind == "intra" else BulkGlobalFactEdgesLabelOnly
    elif neutral_description:
        sig = BulkIntraChunkFactEdgesNeutral if kind == "intra" else BulkGlobalFactEdgesNeutral
    else:
        sig = BulkIntraChunkFactEdges if kind == "intra" else BulkGlobalFactEdges
    return (dspy.ChainOfThought if use_cot else dspy.Predict)(sig)


def _fact_edge_descriptions(edge_pair, neutral_description: bool):
    """(forward, backward) text stored in each edge's `description` attribute. A label-only pair has no
    description, so each direction stores its own label there: retrieval reads `description` for scoring,
    path strings and context, and therefore sees only the short label."""
    if isinstance(edge_pair, LabelOnlyFactEdgePair):
        return edge_pair.label_forward, edge_pair.label_backward
    if neutral_description:
        return edge_pair.description, edge_pair.description
    return edge_pair.description_forward, edge_pair.description_backward


def _edge_extra(edge_pair) -> dict:
    """Extra attributes stored on both directed edges: the relation family of an event-prompt pair."""
    if isinstance(edge_pair, EventFactEdgePair):
        return {"relation_family": str(edge_pair.relation_family).strip().upper()}
    return {}


def _intra_chunk_inputs(graph: Graph, chunk_id, fact_edge_prompt: str) -> dict:
    """Context inputs of an intra-chunk bulk call besides numbered_facts: the chunk summary, plus the chunk's own text
    for the event prompt (BulkIntraChunkFactEdgesEvent)."""
    chunk = graph.chunks[chunk_id]
    inputs = {"chunk_summary": chunk.summary}
    if fact_edge_prompt == "event":
        inputs["chunk_text"] = chunk.text
    return inputs
from graspologic.partition import hierarchical_leiden

def _count_tokens_from(history_idx: int) -> dict:
    """Sums DSPy LM history from history_idx onward into a token-count dict."""
    try:
        entries = dspy.settings.lm.history[history_idx:]
        prompt, completion, cache_hit = 0, 0, 0
        for e in entries:
            p, c, hit = usage_tokens(e.get("usage"))
            prompt += p
            completion += c
            cache_hit += hit
        return {"prompt": prompt, "completion": completion, "total": prompt + completion, "calls": len(entries),
                "cache_hit": cache_hit}
    except Exception:
        return {"prompt": 0, "completion": 0, "total": 0, "calls": 0, "cache_hit": 0}




# 1. Define a pure worker function that ONLY talks to the LLM and returns data









def construct_edges_between_same_chunk_facts_bulk(graph: Graph, use_cot: bool = True, neutral_description: bool = False, same_chunk_min_facts: int = 2, edge_descriptions: bool = True, fact_edge_prompt: str = "default"):
    """
    Constructs intra-chunk fact-to-fact edges using a single LLM call per chunk.

    Instead of one LLM call per fact pair, all facts in a chunk are passed together
    using sequential integer IDs. The LLM identifies relevant pairs and emits all
    edges at once. Integer IDs are then mapped back to UUIDs to build graph edges.

    same_chunk_min_facts — a chunk needs at least this many facts before the LLM call
    is made at all; chunks below the threshold are skipped entirely (no edges, no call).
    """
    bulk_edge_constructor = _bulk_fact_edge_constructor("intra", use_cot, neutral_description, edge_descriptions, fact_edge_prompt)

    for chunk_id in graph.get_chunk_ids():
        fact_ids = graph.get_fact_ids(by_chunk_id=chunk_id)

        if len(fact_ids) < max(2, same_chunk_min_facts):
            continue

        chunk_inputs = _intra_chunk_inputs(graph, chunk_id, fact_edge_prompt)

        # Assign sequential integer IDs and build the numbered facts prompt string.
        # We use ints instead of UUIDs because LLMs are far more reliable at selecting
        # and referencing small integers than full UUID strings.
        int_to_uuid = {}
        numbered_lines = []
        for idx, fact_id in enumerate(fact_ids, start=1):
            int_to_uuid[idx] = fact_id
            numbered_lines.append(f"{idx}: {graph.facts[fact_id].sentence}")

        numbered_facts_str = "\n".join(numbered_lines)

        try:
            result = bulk_edge_constructor(
                **chunk_inputs,
                numbered_facts=numbered_facts_str
            )

            for edge_pair in result.fact_edges:
                src_int = edge_pair.source_id
                tgt_int = edge_pair.target_id

                # Validate that the LLM used real integer IDs from the list
                if src_int not in int_to_uuid or tgt_int not in int_to_uuid:
                    continue
                if src_int == tgt_int:
                    continue
                if edge_pair.relevance_score <= 0.0:
                    continue

                fact_id1 = int_to_uuid[src_int]
                fact_id2 = int_to_uuid[tgt_int]

                if graph.network.has_edge(fact_id1, fact_id2):
                    continue

                desc, desc_back = _fact_edge_descriptions(edge_pair, neutral_description)

                graph.network.add_edge(
                    fact_id1, fact_id2,
                    description=desc,
                    label=edge_pair.label_forward,
                    edge_type="fact_fact",
                    edge_sub_type="same_chunk",
                    score=edge_pair.relevance_score,
                    **_edge_extra(edge_pair)
                )
                graph.network.add_edge(
                    fact_id2, fact_id1,
                    description=desc_back,
                    label=edge_pair.label_backward,
                    edge_type="fact_fact",
                    edge_sub_type="same_chunk",
                    score=edge_pair.relevance_score,
                    **_edge_extra(edge_pair)
                )

        except Exception as e:
            print(f"Error constructing bulk intra-chunk edges for chunk {chunk_id}: {e}")

    print("Finished constructing bulk intra-chunk fact edges.")


def _bulk_same_chunk_fact_worker(bulk_edge_constructor, chunk_id, chunk_inputs, int_to_uuid, numbered_facts_str):
    """Worker: runs one bulk LLM call for a chunk and returns raw results for the main thread to add to the graph.
    chunk_inputs: _intra_chunk_inputs (chunk_summary, and chunk_text for the event prompt)."""
    result = bulk_edge_constructor(
        **chunk_inputs,
        numbered_facts=numbered_facts_str
    )
    return chunk_id, int_to_uuid, result


def construct_edges_between_same_chunk_facts_bulk_parallel(graph: Graph, max_workers: int = 10, use_cot: bool = True, neutral_description: bool = False, same_chunk_min_facts: int = 2, edge_descriptions: bool = True, fact_edge_prompt: str = "default"):
    """
    Parallel version of construct_edges_between_same_chunk_facts_bulk.
    One LLM call per chunk, all calls run in parallel across a thread pool.

    same_chunk_min_facts — a chunk needs at least this many facts before the LLM call
    is made at all; chunks below the threshold are skipped entirely (no edges, no call).
    """
    bulk_edge_constructor = _bulk_fact_edge_constructor("intra", use_cot, neutral_description, edge_descriptions, fact_edge_prompt)

    tasks = []
    for chunk_id in graph.get_chunk_ids():
        fact_ids = graph.get_fact_ids(by_chunk_id=chunk_id)
        if len(fact_ids) < max(2, same_chunk_min_facts):
            continue

        chunk_inputs = _intra_chunk_inputs(graph, chunk_id, fact_edge_prompt)
        int_to_uuid = {}
        numbered_lines = []
        for idx, fact_id in enumerate(fact_ids, start=1):
            int_to_uuid[idx] = fact_id
            numbered_lines.append(f"{idx}: {graph.facts[fact_id].sentence}")

        tasks.append((chunk_id, chunk_inputs, int_to_uuid, "\n".join(numbered_lines)))

    print(f"Executing {len(tasks)} bulk intra-chunk fact evaluations across {max_workers} threads...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_bulk_same_chunk_fact_worker, bulk_edge_constructor, t[0], t[1], t[2], t[3]): t
            for t in tasks
        }

        for future in as_completed(futures):
            try:
                chunk_id, int_to_uuid, result = future.result()

                for edge_pair in result.fact_edges:
                    src_int = edge_pair.source_id
                    tgt_int = edge_pair.target_id

                    if src_int not in int_to_uuid or tgt_int not in int_to_uuid:
                        continue
                    if src_int == tgt_int:
                        continue
                    if edge_pair.relevance_score <= 0.0:
                        continue

                    fact_id1 = int_to_uuid[src_int]
                    fact_id2 = int_to_uuid[tgt_int]

                    if graph.network.has_edge(fact_id1, fact_id2):
                        continue

                    desc, desc_back = _fact_edge_descriptions(edge_pair, neutral_description)

                    graph.network.add_edge(
                        fact_id1, fact_id2,
                        description=desc,
                        label=edge_pair.label_forward,
                        edge_type="fact_fact",
                        edge_sub_type="same_chunk",
                        score=edge_pair.relevance_score,
                        **_edge_extra(edge_pair)
                    )
                    graph.network.add_edge(
                        fact_id2, fact_id1,
                        description=desc_back,
                        label=edge_pair.label_backward,
                        edge_type="fact_fact",
                        edge_sub_type="same_chunk",
                        score=edge_pair.relevance_score,
                        **_edge_extra(edge_pair)
                    )
            except Exception as e:
                print(f"Error processing bulk intra-chunk fact edge: {e}")

    print("Finished constructing bulk intra-chunk fact edges (parallel).")








# --- MAIN PARALLELIZED FUNCTION ---






# --- MAIN PARALLELIZED FUNCTION ---








def _get_parent_chunk_summary(graph: Graph, fact_id: str) -> str:
    """The summary of a fact's parent chunk (found through its incoming chunk_fact edge), as LLM context."""
    for source_node, _target_node, edge_data in graph.network.in_edges(fact_id, data=True):
        if edge_data.get("edge_type") == "chunk_fact":
            return graph.chunks[source_node].summary
    return ""


def _bulk_global_fact_worker(bulk_edge_constructor, int_to_uuid, numbered_facts_str):
    """Worker: runs one bulk LLM call for a cluster and returns raw results for the main thread."""
    result = bulk_edge_constructor(numbered_facts=numbered_facts_str)
    return int_to_uuid, result


def _run_bulk_fact_edge_llm(graph: Graph, indexed_cluster_lists: list, bulk_edge_constructor, max_workers: int, edge_sub_type: str, neutral_description: bool = False, on_cluster_done=None):
    """
    Prepares numbered-fact strings for each cluster, runs parallel LLM calls,
    then writes the resulting edges to the graph (main thread only).
    Called once per clustering round with the appropriate edge_sub_type tag.

    indexed_cluster_lists — list of (cluster_idx, cluster_fact_ids) tuples. cluster_idx is the
    cluster's position in the round's full cluster list (stable across resumes) so a checkpoint
    can track exactly which clusters are done; callers filter out already-completed clusters
    before calling this.

    on_cluster_done(cluster_idx, edges) — optional callback invoked in the main thread right
    after a cluster's edges are written into `graph`. `edges` is the list of
    {source, target, description, label, edge_type, edge_sub_type, score} dicts just added
    (both directions), possibly empty if the cluster produced no qualifying edges. Called
    exactly once per successfully-processed cluster (not called if the LLM call itself raised).
    """
    tasks = []
    for cluster_idx, cluster_fact_ids in indexed_cluster_lists:
        int_to_uuid = {}
        numbered_lines = []
        for idx, fact_id in enumerate(cluster_fact_ids, start=1):
            int_to_uuid[idx] = fact_id
            fact = graph.facts[fact_id]
            chunk_summary = _get_parent_chunk_summary(graph, fact_id)
            numbered_lines.append(f"{idx}: [Context: {chunk_summary}] {fact.sentence}")
        tasks.append((cluster_idx, int_to_uuid, "\n".join(numbered_lines)))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_bulk_global_fact_worker, bulk_edge_constructor, t[1], t[2]): t[0]
            for t in tasks
        }
        for future in as_completed(futures):
            cluster_idx = futures[future]
            try:
                int_to_uuid, result = future.result()
                new_edges = []
                for edge_pair in result.fact_edges:
                    src_int = edge_pair.source_id
                    tgt_int = edge_pair.target_id

                    if src_int not in int_to_uuid or tgt_int not in int_to_uuid:
                        continue
                    if src_int == tgt_int:
                        continue
                    if edge_pair.relevance_score <= 0.0:
                        continue

                    fact_id1 = int_to_uuid[src_int]
                    fact_id2 = int_to_uuid[tgt_int]

                    if graph.network.has_edge(fact_id1, fact_id2):
                        continue

                    desc, desc_back = _fact_edge_descriptions(edge_pair, neutral_description)
                    # event prompt (EventFactEdgePair): the relation family travels with both directed edges
                    extra = _edge_extra(edge_pair)

                    graph.network.add_edge(
                        fact_id1, fact_id2,
                        description=desc,
                        label=edge_pair.label_forward,
                        edge_type="fact_fact",
                        edge_sub_type=edge_sub_type,
                        score=edge_pair.relevance_score,
                        **extra
                    )
                    graph.network.add_edge(
                        fact_id2, fact_id1,
                        description=desc_back,
                        label=edge_pair.label_backward,
                        edge_type="fact_fact",
                        edge_sub_type=edge_sub_type,
                        score=edge_pair.relevance_score,
                        **extra
                    )
                    new_edges.append({"source": fact_id1, "target": fact_id2, "description": desc,
                                       "label": edge_pair.label_forward, "edge_type": "fact_fact",
                                       "edge_sub_type": edge_sub_type, "score": edge_pair.relevance_score, **extra})
                    new_edges.append({"source": fact_id2, "target": fact_id1, "description": desc_back,
                                       "label": edge_pair.label_backward, "edge_type": "fact_fact",
                                       "edge_sub_type": edge_sub_type, "score": edge_pair.relevance_score, **extra})

                if on_cluster_done is not None:
                    on_cluster_done(cluster_idx, new_edges)
            except Exception as e:
                print(f"Error processing clustered fact edge batch (cluster {cluster_idx}): {e}")


_ZERO_TOKENS = {"prompt": 0, "completion": 0, "total": 0, "calls": 0, "cache_hit": 0}


def _add_tokens(a: dict, b: dict) -> dict:
    return {
        "prompt":     a.get("prompt", 0) + b.get("prompt", 0),
        "completion": a.get("completion", 0) + b.get("completion", 0),
        "total":      a.get("total", 0) + b.get("total", 0),
        "calls":      a.get("calls", 0) + b.get("calls", 0),
        "cache_hit":  a.get("cache_hit", 0) + b.get("cache_hit", 0),
    }


def _process_clustered_round(graph: Graph, round_num: int, use_round: bool, sim_graph_builder, max_cluster_size: int,
                             bulk_edge_constructor, max_workers: int, edge_sub_type: str, neutral_description: bool,
                             checkpoint, cluster_fn=None) -> dict:
    """
    Handles checkpoint resume/replay + Leiden clustering + bulk LLM edge construction for one
    round of clustered global fact-edge construction. Returns this round's cumulative token
    dict (summed across every resume attempt, via `checkpoint`; just this attempt's tokens
    when checkpointing is disabled).

    sim_graph_builder — zero-arg callable that builds and returns this round's nx.Graph of
    candidate fact pairs. Only called when there's no already-saved cluster list to resume from,
    so a resumed run never rebuilds embeddings/re-clusters work that's already been done.

    cluster_fn — optional callable(sim_graph) -> list of cluster fact-id lists, replacing the
    built-in grouping below (used by fast_fact_clustering, edge_mode="clustered_fast"). None =
    the built-in grouping, which keeps every hierarchical_leiden level (see fast_fact_clustering).
    """
    if not use_round:
        print(f"Round {round_num}: skipped (use_round{round_num}=False).")
        return dict(_ZERO_TOKENS)

    if checkpoint.round_status(round_num) == "done":
        print(f"Round {round_num}: already complete (checkpoint) — replaying saved edges.")
        checkpoint.replay_round_edges(round_num, graph)
        return checkpoint.round_tokens(round_num)

    before_tokens = checkpoint.round_tokens(round_num)

    if checkpoint.has_cluster_list(round_num):
        cluster_lists = checkpoint.load_cluster_list(round_num)
        completed_before = checkpoint.completed_cluster_indices(round_num)
        checkpoint.replay_round_edges(round_num, graph)
        print(f"Round {round_num}: resuming — {len(completed_before)}/{len(cluster_lists)} clusters already done.")
    else:
        print(f"Round {round_num}: building similarity graph...")
        sim_graph = sim_graph_builder()
        print(f"Round {round_num} sim graph: {sim_graph.number_of_edges()} candidate edges.")
        if cluster_fn is not None:
            cluster_lists = cluster_fn(sim_graph)
        elif sim_graph.number_of_edges() > 0:
            clusters_raw = hierarchical_leiden(sim_graph, max_cluster_size=max_cluster_size, random_seed=42)
            clusters = {}
            for node in clusters_raw:
                clusters.setdefault(node.cluster, []).append(node.node)
            cluster_lists = [c for c in clusters.values() if len(c) > 1]
        else:
            cluster_lists = []
        checkpoint.save_cluster_list(round_num, cluster_lists)
        completed_before = set()
        print(f"Round {round_num}: {len(cluster_lists)} clusters → running {len(cluster_lists)} bulk LLM calls...")

    remaining = [(i, c) for i, c in enumerate(cluster_lists) if i not in completed_before]
    if not remaining:
        checkpoint.mark_round_done(round_num)
        return before_tokens

    succeeded = set()

    def _on_done(cluster_idx, edges):
        succeeded.add(cluster_idx)
        checkpoint.append_cluster_edges(round_num, cluster_idx, edges)

    try: _h = len(dspy.settings.lm.history)
    except Exception: _h = 0
    _t0 = time.time()
    _run_bulk_fact_edge_llm(graph, remaining, bulk_edge_constructor, max_workers,
                            edge_sub_type=edge_sub_type, neutral_description=neutral_description,
                            on_cluster_done=_on_done)
    attempt_tokens = _count_tokens_from(_h)
    attempt_time = time.time() - _t0
    checkpoint.add_round_tokens_and_time(round_num, attempt_tokens, attempt_time)

    total_done = completed_before | succeeded
    if total_done == set(range(len(cluster_lists))):
        checkpoint.mark_round_done(round_num)
    else:
        missing = set(range(len(cluster_lists))) - total_done
        print(f"Round {round_num}: {len(missing)} cluster(s) failed this attempt and will be retried on next resume.")

    return _add_tokens(before_tokens, attempt_tokens)


def construct_edges_between_facts_clustered(graph: Graph, top_k: int = 10, skip_percent: float = 0.33,
                                            max_cluster_size: int = 20, max_workers: int = 10,
                                            retrieval_model: str = "all-MiniLM-L6-v2", use_cot: bool = True,
                                            neutral_description: bool = False,
                                            use_round1: bool = True, use_round2: bool = True, use_round3: bool = True,
                                            edge_descriptions: bool = True, fact_edge_prompt: str = "default",
                                            r1_weights: dict = None, r2_weights: dict = None, r3_weights: dict = None,
                                            questions_mode: str = "follow_up_to_answered",
                                            checkpoint=None):
    """
    Builds global cross-chunk fact-to-fact edges using 3 rounds of similarity-based
    clustering followed by one bulk LLM call per cluster.

    Round 1 — Entity-based (entity_weight=1.0, all others=0.0), top_k:
        Groups facts that share the same entities/roles. Catches relationships
        between facts about the same real-world objects even if their sentences
        are phrased very differently.

    Round 2 — Balanced top_k (entity_weight=0.5, all others=1.0):
        Groups the most semantically similar facts overall, with entity overlap
        as a secondary signal.

    Round 3 — Balanced middle_k (same weights as Round 2, skipping the top percentile):
        Discovery round — surfaces less-obvious but still meaningful connections
        that were too similar to appear in the top-k (already covered) but still
        sit in a plausible relevance band.

    In all rounds, fact pairs that already share an edge in the graph are excluded
    from the similarity graph before clustering, so no LLM work is duplicated.
    Edges created in Round 1 are naturally excluded by Round 2's filter, and so on.
    """
    checkpoint = checkpoint if checkpoint is not None else ConstructionCheckpoint(None)

    print("Initializing fact embeddings for clustered edge construction...")
    graph.init_fact_embeddings(retrieval_model=retrieval_model)

    # Resolve per-round similarity weights, falling back to documented defaults.
    _r1 = r1_weights or {"sent": 0.0, "topic": 0.0, "entity": 1.0, "questions": 0.0}
    _r2 = r2_weights or {"sent": 1.0, "topic": 1.0, "entity": 1.0, "questions": 1.0}
    _r3 = r3_weights or {"sent": 1.0, "topic": 1.0, "entity": 0.5, "questions": 1.0}

    fact_ids = graph.get_fact_ids()
    bulk_edge_constructor = _bulk_fact_edge_constructor("global", use_cot, neutral_description, edge_descriptions, fact_edge_prompt)

    def _build_sim_graph(weights: dict, strategy: str, skip_top_percent: float = 0.0) -> nx.Graph:
        sim_graph = nx.Graph()
        sim_graph.add_nodes_from(fact_ids)
        for fact_id in fact_ids:
            all_ids, all_scores = graph.get_relevant_seeds_by_id(
                fact_id, get_all=True,
                sent_weight=weights["sent"], topic_weight=weights["topic"],
                entity_weight=weights["entity"], questions_weight=weights["questions"],
                questions_mode=questions_mode
            )
            id_to_score = dict(zip(all_ids, all_scores))
            if strategy == "top_k":
                candidate_ids, _ = sample_results(all_ids, all_scores, strategy="top_k", k=top_k)
            else:
                candidate_ids, _ = sample_results(all_ids, all_scores, strategy="skip_percent_top_k",
                                                  k=top_k, skip_top_percent=skip_top_percent)
            for candidate_id in candidate_ids:
                if candidate_id == fact_id:
                    continue
                if graph.network.has_edge(fact_id, candidate_id):
                    continue
                sim_graph.add_edge(fact_id, candidate_id, weight=max(0.01, float(id_to_score[candidate_id])))
        return sim_graph

    # ==========================================
    # ROUND 1: Entity-based clustering → top_k
    # ==========================================
    round1_tokens = _process_clustered_round(
        graph, 1, use_round1,
        sim_graph_builder=lambda: _build_sim_graph(_r1, "top_k"),
        max_cluster_size=max_cluster_size, bulk_edge_constructor=bulk_edge_constructor,
        max_workers=max_workers, edge_sub_type="global_topk_entity_cluster",
        neutral_description=neutral_description, checkpoint=checkpoint,
    )

    # ==========================================
    # ROUND 2: Balanced top-k clustering
    # ==========================================
    round2_tokens = _process_clustered_round(
        graph, 2, use_round2,
        sim_graph_builder=lambda: _build_sim_graph(_r2, "top_k"),
        max_cluster_size=max_cluster_size, bulk_edge_constructor=bulk_edge_constructor,
        max_workers=max_workers, edge_sub_type="global_topk_balanced",
        neutral_description=neutral_description, checkpoint=checkpoint,
    )

    # ==========================================
    # ROUND 3: Balanced middle-k clustering (skips top percentile)
    # ==========================================
    round3_tokens = _process_clustered_round(
        graph, 3, use_round3,
        sim_graph_builder=lambda: _build_sim_graph(_r3, "middle_k", skip_top_percent=skip_percent),
        max_cluster_size=max_cluster_size, bulk_edge_constructor=bulk_edge_constructor,
        max_workers=max_workers, edge_sub_type="global_middlek_balanced_cluster",
        neutral_description=neutral_description, checkpoint=checkpoint,
    )

    print("Finished constructing clustered global fact edges.")
    return {"round1": round1_tokens, "round2": round2_tokens, "round3": round3_tokens}


def construct_initial_edges(graph: Graph, threshold: float = 0.5, max_workers: int = 10, use_cot: bool = True, neutral_description: bool = False, same_chunk_min_facts: int = 2, edge_descriptions: bool = True, fact_edge_prompt: str = "default"):
    """Intra-chunk fact->fact edges (one bulk LLM call per chunk). Entity->fact edges are created during fact
    extraction (BatchedFactAssembler) from the entity_relation_labels stored on each Fact."""
    if max_workers == 1:
        construct_edges_between_same_chunk_facts_bulk(graph, use_cot=use_cot, neutral_description=neutral_description, same_chunk_min_facts=same_chunk_min_facts, edge_descriptions=edge_descriptions, fact_edge_prompt=fact_edge_prompt)
    else:
        construct_edges_between_same_chunk_facts_bulk_parallel(graph, max_workers=max_workers, use_cot=use_cot, neutral_description=neutral_description, same_chunk_min_facts=same_chunk_min_facts, edge_descriptions=edge_descriptions, fact_edge_prompt=fact_edge_prompt)


def construct_edges_during_merge(graph: Graph, threshold: float = 0.5, max_workers: int = 10, mode: str = "clustered", top_k: int = 10, use_cot: bool = True, neutral_description: bool = False,
                                 use_round1: bool = True, use_round2: bool = True, use_round3: bool = True,
                                 clustered_max_facts_per_prompt: int = 20,
                                 retrieval_model: str = "all-MiniLM-L6-v2",
                                 r1_weights: dict = None, r2_weights: dict = None, r3_weights: dict = None,
                                 questions_mode: str = "follow_up_to_answered",
                                 middle_k_skip_percent: float = 0.33,
                                 checkpoint=None, edge_descriptions: bool = True, fact_edge_prompt: str = "default",
                                 fast_cluster_options: dict = None):
    if mode == "clustered_fast":
        # fast_fact_clustering: batched GPU similarity graph, cached embeddings, Leiden-level choice.
        # fast_cluster_options: {"embed_loader", "leiden_levels", "leiden_passes"} (module defaults when absent).
        from nkg.index.construction.fast_fact_clustering import construct_edges_between_facts_clustered_fast
        return construct_edges_between_facts_clustered_fast(graph, top_k=top_k, skip_percent=middle_k_skip_percent, max_workers=max_workers, use_cot=use_cot, neutral_description=neutral_description, edge_descriptions=edge_descriptions, fact_edge_prompt=fact_edge_prompt,
                                                            use_round1=use_round1, use_round2=use_round2, use_round3=use_round3,
                                                            max_cluster_size=clustered_max_facts_per_prompt,
                                                            retrieval_model=retrieval_model,
                                                            r1_weights=r1_weights, r2_weights=r2_weights, r3_weights=r3_weights,
                                                            questions_mode=questions_mode,
                                                            checkpoint=checkpoint, **(fast_cluster_options or {}))
    if mode == "clustered":
        # One LLM call per cluster instead of one per fact-pair.
        # max_workers controls parallel LLM calls across clusters within each round.
        return construct_edges_between_facts_clustered(graph, top_k=top_k, skip_percent=middle_k_skip_percent, max_workers=max_workers, use_cot=use_cot, neutral_description=neutral_description, edge_descriptions=edge_descriptions, fact_edge_prompt=fact_edge_prompt,
                                                       use_round1=use_round1, use_round2=use_round2, use_round3=use_round3,
                                                       max_cluster_size=clustered_max_facts_per_prompt,
                                                       retrieval_model=retrieval_model,
                                                       r1_weights=r1_weights, r2_weights=r2_weights, r3_weights=r3_weights,
                                                       questions_mode=questions_mode,
                                                       checkpoint=checkpoint)
    raise ValueError(f"edge mode must be 'clustered' or 'clustered_fast', got {mode!r}")
