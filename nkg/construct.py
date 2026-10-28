#!/usr/bin/env python3
"""Build one UDH index (kg.graphml) over a corpus.

    python -m nkg.construct --corpus-file <run>/data/corpus.jsonl --run-dir <run> --input-mode prepared_chunks

Input: a JSONL corpus, one {"id", "title", "paragraph"} per line (evaluation_scripts/unh/main.py --step prepare writes
it). Each document becomes the text "Title: <title>\\n\\n<paragraph>"; --input-mode decides how it is chunked:
  prepared_chunks  each document is one chunk, not split further (HotpotQA, MuSiQue: one passage = one chunk)
  per_document     each document is chunked on its own (600/50 tokens), no chunk crosses documents (G-Medical)
  concat_file      all documents are joined into one text (separator "\\n\\n---\\n\\n") and chunked (600/50) (contracts)

Every other construction setting is fixed to the paper's values (CONSTRUCTION_ARGS below); the options here are the
ones that differ between the paper's datasets (see evaluation_scripts/unh/main.py, DATASET_CONSTRUCTION) plus
execution knobs. Outputs in --run-dir: kg.graphml, construction_results.json (configuration, graph size, LLM token
usage per stage), construction_config.json. Construction is checkpointed under --run-dir/checkpoint/; re-running the
same command resumes it.

The LLM is DeepSeek (provider "deepseek", key from DEEPSEEK_API_KEY) when --api-base is api.deepseek.com, otherwise
any OpenAI-compatible server (e.g. vLLM; key from LLM_API_KEY, default "EMPTY").
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

from nkg.index.construction.build_index import build_index
from nkg.index.construction.checkpointing import ConstructionCheckpoint
from nkg.utils.config import configure_dspy

# The paper's construction settings shared by every dataset (Appendix, Table A1).
CONSTRUCTION_ARGS = {
    "chunk_size": 600,
    "chunk_overlap": 50,
    "chunk_tokenizer_model": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
    "fact_batch_size": 15,                 # facts annotated per call
    "same_chunk_min_facts": 3,             # minimum facts for the intra-chunk relation call
    "edge_threshold": 0.5,
    "top_k": 12,                           # candidate neighbours per fact for cross-chunk communities
    "middle_k_skip_percent": 0.33,
    "clustered_round1": False,
    "clustered_round2": True,
    "clustered_round3": False,
    "use_questions_weight": True,
    "questions_mode": "follow_up_to_answered",
    "global_edge_retrieval_model": "voyageai/voyage-4-nano",
    "edge_descriptions": False,            # label-only fact->fact edges
    "neutral_description": True,
    "concise_summary": True,
    "combined_entity_extraction": True,
    "full_entity_profile": False,
    "generate_entity_relation_labels": False,
    "use_cot": False,
    # entity resolution
    "run_deduplication": True,
    "dedup_model_name": "voyageai/voyage-4-nano",
    "dedup_iterations": 1,
    "dedup_max_cluster_size": 72,
    "dedup_run_exact": True,
    "dedup_run_semhash": True,
    "dedup_semhash_threshold": 0.95,
    "dedup_run_llm": True,
    "dedup_similarity_threshold": 0.5,
    "dedup_sim_weights": {"name": 0.6, "role": 0.3, "type": 0.1},
    # fact consolidation
    "run_fact_dedup": True,
    "fact_dedup_top_k": 20,
    "fact_dedup_cluster_size": 30,
    # clustered_fast community detection
    "fast_cluster_embed_loader": "remote_code",
    "fast_cluster_leiden_levels": "final",
    "fast_cluster_leiden_passes": 1,
    # LLM calls
    "construction_max_tokens": 100000,
    "num_retries": 1,
    "temperature": None,                   # not sent: the provider's default
}

CONFIG_FILE = "construction_config.json"
RESULTS_FILE = "construction_results.json"
KG_FILE = "kg.graphml"


def load_corpus(path: str) -> list:
    """Documents of a JSONL corpus as "Title: <title>\\n\\n<paragraph>", in id order."""
    docs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                docs[d["id"]] = f"Title: {d['title']}\n\n{d['paragraph']}"
    return [docs[k] for k in sorted(docs)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus-file", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--input-mode", required=True, choices=["prepared_chunks", "per_document", "concat_file"])
    ap.add_argument("--edge-mode", default="clustered_fast", choices=["clustered", "clustered_fast"],
                    help="cross-chunk fact communities: clustered_fast (batched GPU similarity graph + Leiden; "
                         "MuSiQue, G-Medical, contracts) or clustered (the first implementation; HotpotQA)")
    ap.add_argument("--clustered-max-facts-per-prompt", type=int, default=45,
                    help="maximum fact-community size M (HotpotQA / MuSiQue 50, G-Medical / contracts 45)")
    ap.add_argument("--fast-cluster-min-size", type=int, default=20,
                    help="clustered_fast: minimum fact-community size M_min, smaller communities are merged into "
                         "their most connected neighbour (MuSiQue 0 = off, G-Medical / contracts 20)")
    ap.add_argument("--fact-prompt", default="detailed", choices=["default", "detailed"],
                    help="fact extraction prompt (HotpotQA / MuSiQue default, G-Medical / contracts detailed)")
    ap.add_argument("--fact-edge-prompt", default="event", choices=["default", "event"],
                    help="fact->fact relation prompt (HotpotQA / MuSiQue default, G-Medical / contracts event)")
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", "deepseek-v4-flash"))
    ap.add_argument("--api-base", default=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"))
    ap.add_argument("--max-workers", type=int, default=64, help="chunks extracted at once")
    ap.add_argument("--max-sub-workers", type=int, default=64,
                    help="LLM calls at once in the deduplication and cross-chunk relation stages")
    ap.add_argument("--force-rebuild", action="store_true", help="discard the checkpoint and start over")
    args = ap.parse_args()

    os.makedirs(args.run_dir, exist_ok=True)
    results_path = os.path.join(args.run_dir, RESULTS_FILE)
    config_path = os.path.join(args.run_dir, CONFIG_FILE)
    checkpoint_dir = os.path.join(args.run_dir, "checkpoint")
    if os.path.exists(results_path) and not args.force_rebuild:
        print(f"[construct] {results_path} exists: the index is built (--force-rebuild to rebuild)")
        return

    config = {**CONSTRUCTION_ARGS, "input_mode": args.input_mode, "edge_mode": args.edge_mode,
              "clustered_max_facts_per_prompt": args.clustered_max_facts_per_prompt,
              "fast_cluster_min_size": args.fast_cluster_min_size, "fact_prompt": args.fact_prompt,
              "fact_edge_prompt": args.fact_edge_prompt, "global_fact_edge_prompt": args.fact_edge_prompt,
              "model_name": args.model}
    if args.force_rebuild:
        ConstructionCheckpoint(checkpoint_dir).wipe()
    elif os.path.exists(config_path):
        saved = json.load(open(config_path))["args"]
        diff = {k: (saved.get(k), v) for k, v in config.items() if saved.get(k) != v}
        if diff:
            sys.exit(f"[construct] {args.run_dir} holds a checkpoint built with other settings {diff}; "
                     f"pass the same settings to resume, or --force-rebuild to start over")
        print(f"[construct] resuming from {checkpoint_dir}")
    json.dump({"saved_at": datetime.now().isoformat(), "args": config}, open(config_path, "w"), indent=2)

    deepseek = "deepseek.com" in args.api_base
    api_key = os.environ.get("DEEPSEEK_API_KEY" if deepseek else "LLM_API_KEY", "" if deepseek else "EMPTY")
    if deepseek and not api_key:
        sys.exit("[construct] export DEEPSEEK_API_KEY first")
    configure_dspy(model=f"{'deepseek' if deepseek else 'openai'}/{args.model}", api_base=args.api_base,
                   api_key=api_key, max_tokens=config["construction_max_tokens"], cache=False,
                   num_retries=config["num_retries"], temperature=config["temperature"])

    docs = load_corpus(args.corpus_file)
    if args.input_mode == "concat_file":
        filepath, chunks = os.path.join(args.run_dir, "corpus.txt"), None
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n\n---\n\n".join(docs))
    else:
        filepath, chunks = None, docs
    total_chars = sum(len(d) for d in docs)
    print(f"[construct] {len(docs):,} documents ({total_chars:,} chars), input mode {args.input_mode}, "
          f"edge mode {args.edge_mode}, checkpoint {checkpoint_dir}", flush=True)

    c = config
    t0 = time.time()
    graph, token_breakdown = build_index(
        filepath=filepath, chunks=chunks,
        chunk_list_inputs=args.input_mode == "per_document",  # per_document: build_index chunks each document
        model_name=c["chunk_tokenizer_model"], max_workers=args.max_workers, chunk_size=c["chunk_size"],
        overlap=c["chunk_overlap"], edge_threshold=c["edge_threshold"], top_k=c["top_k"],
        max_sub_workers=args.max_sub_workers, fact_batch_size=c["fact_batch_size"], edge_mode=c["edge_mode"],
        run_deduplication=c["run_deduplication"], dedup_model_name=c["dedup_model_name"],
        dedup_iterations=c["dedup_iterations"], dedup_max_cluster_size=c["dedup_max_cluster_size"],
        use_cot=c["use_cot"], neutral_description=c["neutral_description"], concise_summary=c["concise_summary"],
        clustered_round1=c["clustered_round1"], clustered_round2=c["clustered_round2"],
        clustered_round3=c["clustered_round3"], clustered_max_facts_per_prompt=c["clustered_max_facts_per_prompt"],
        middle_k_skip_percent=c["middle_k_skip_percent"], retrieval_model_name=c["global_edge_retrieval_model"],
        construction_r2_weights={"sent": 1.0, "topic": 1.0, "entity": 1.0,
                                 "questions": 1.0 if c["use_questions_weight"] else 0.0},
        construction_r3_weights={"sent": 1.0, "topic": 1.0, "entity": 1.0,
                                 "questions": 1.0 if c["use_questions_weight"] else 0.0},
        questions_mode=c["questions_mode"], dedup_run_exact=c["dedup_run_exact"],
        dedup_run_semhash=c["dedup_run_semhash"], dedup_semhash_threshold=c["dedup_semhash_threshold"],
        dedup_run_llm=c["dedup_run_llm"], dedup_similarity_threshold=c["dedup_similarity_threshold"],
        combined_entity_extraction=c["combined_entity_extraction"], full_entity_profile=c["full_entity_profile"],
        generate_entity_relation_labels=c["generate_entity_relation_labels"],
        dedup_sim_weights=c["dedup_sim_weights"], same_chunk_min_facts=c["same_chunk_min_facts"],
        edge_descriptions=c["edge_descriptions"], run_fact_dedup=c["run_fact_dedup"],
        fact_dedup_top_k=c["fact_dedup_top_k"], fact_dedup_cluster_size=c["fact_dedup_cluster_size"],
        fast_cluster_embed_loader=c["fast_cluster_embed_loader"],
        fast_cluster_leiden_levels=c["fast_cluster_leiden_levels"],
        fast_cluster_leiden_passes=c["fast_cluster_leiden_passes"], fast_cluster_min_size=c["fast_cluster_min_size"],
        fact_prompt=c["fact_prompt"], fact_edge_prompt=c["fact_edge_prompt"],
        global_fact_edge_prompt=c["global_fact_edge_prompt"], checkpoint_dir=checkpoint_dir,
    )
    ckpt = ConstructionCheckpoint(checkpoint_dir)
    # wall time over every resumed attempt for this run dir
    seconds = round(ckpt.extraction_time_total() + ckpt.dedup_time_total()
                    + sum(ckpt.round_time(n) for n in (1, 2, 3)), 2) or round(time.time() - t0, 2)
    kg_path = os.path.join(args.run_dir, KG_FILE)
    graph.export_graph(kg_path)
    total = token_breakdown.get("total", {})
    results = {
        "system": "UDH",
        "timestamp": datetime.now().isoformat(),
        "metadata": {"corpus_documents": len(docs), "model": args.model,
                     "provider": "deepseek" if deepseek else "openai_compatible",
                     "construction_args": {**config, "max_workers": args.max_workers,
                                           "max_sub_workers": args.max_sub_workers}},
        "construction": {
            "total_corpus_chars": total_chars,
            "n_nodes": graph.network.number_of_nodes(),
            "n_edges": graph.network.number_of_edges(),
            "n_entities": len(graph.entities),
            "n_facts": len(graph.facts),
            "execution_time_sec": seconds,
            "kg_size_bytes": os.path.getsize(kg_path),
            "token_breakdown": token_breakdown,
            "llm_prompt_tokens": total.get("prompt", 0),
            "llm_completion_tokens": total.get("completion", 0),
            "llm_total_tokens": total.get("total", 0),
            "llm_api_calls": total.get("calls", 0),
            "llm_prompt_cache_hit_tokens": total.get("cache_hit", 0),
        },
    }
    json.dump(results, open(results_path, "w", encoding="utf-8"), indent=2, default=str)
    print(f"[construct] done in {seconds}s: {results['construction']['n_facts']:,} facts, "
          f"{results['construction']['n_entities']:,} entities, {total.get('total', 0):,} LLM tokens -> {kg_path}")


if __name__ == "__main__":
    main()
