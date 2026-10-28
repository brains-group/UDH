#!/usr/bin/env python3
"""UDH (package nkg) on one dataset: corpus preparation, index construction and retrieval. The retrieved contexts then
go through the shared answer step and scorer like every other system (run.sh, run_gmedical.sh, run_contracts.sh).

    python evaluation_scripts/unh/main.py --dataset hotpotqa --step all --view full,hypergraph,discourse
    python evaluation_scripts/unh/main.py --dataset hotpotqa --step retrieve --view full --ppr-passages 5 --tag hyb_p5

Steps (in --out-dir, default evaluation_scripts/unh/outputs/<dataset>):
  prepare    the corpus -> data/corpus.jsonl ({id, title, paragraph}; ids p000000.. keep the corpus order)
               hotpotqa / musique  evaluation_data/<ds>/<ds>_corpus.json, one passage per document
               gmedical            <inputs>/documents.json (the medical corpus as one document)
               contracts           <inputs>/contract.txt (one contract = one document)
  construct  python -m nkg.construct with the dataset's construction settings (DATASET_CONSTRUCTION) -> kg.graphml,
             construction_results.json (configuration, graph size, LLM token usage). Resumable from checkpoint/.
  retrieve   nkg.retrieval.ppr_engine.PPRRetriever (no LLM call), the paper's setting (DEFAULT_PPR_ARGS), for each
             --view; one context per question -> contexts[_<view>][_<tag>].json plus retrieval_report_<view>[_<tag>].json
               full        every edge type                          UDH    (answer step system "unh")
               hypergraph  entity<->fact edges only (participation)  UDH-H  ("unh_hypergraph")
               discourse   fact->fact edges only (dependency)        UDH-D  ("unh_discourse")
             --ppr-passages 5 --tag hyb_p5: the same paths plus the five chunks with the highest PPR scores (UDH+5p,
             "unh_hyb_p5"), from the same index.
  all        prepare + construct + retrieve

Construction needs DEEPSEEK_API_KEY (common/env.sh); retrieval needs no API key. voyage-4-nano and the cross-encoder
run in-process (GPU if available); their embeddings are cached next to the graph.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA_ROOT = REPO / "evaluation_data"
sys.path.insert(0, str(REPO))

VIEWS = {"full": ("all", "contexts.json"), "hypergraph": ("hypergraph", "contexts_hypergraph.json"),
         "discourse": ("discourse", "contexts_discourse.json")}

# Construction settings that differ between the paper's datasets (everything else: nkg.construct.CONSTRUCTION_ARGS).
#   input_mode     how documents become chunks (nkg/construct.py)
#   edge_mode      cross-chunk fact communities: "clustered" (the first implementation, HotpotQA) or "clustered_fast"
#                  (batched GPU similarity graph + Leiden; the later runs)
#   max / min      maximum fact-community size M / minimum size M_min (0 = no merging of small communities)
#   fact_prompt    fact extraction prompt; fact_edge_prompt: fact->fact relation prompt (intra- and cross-chunk)
DATASET_CONSTRUCTION = {
    "hotpotqa": {"input_mode": "prepared_chunks", "edge_mode": "clustered", "max": 50, "min": 0,
                 "fact_prompt": "default", "fact_edge_prompt": "default"},
    "musique": {"input_mode": "prepared_chunks", "edge_mode": "clustered_fast", "max": 50, "min": 0,
                "fact_prompt": "default", "fact_edge_prompt": "default"},
    "gmedical": {"input_mode": "per_document", "edge_mode": "clustered_fast", "max": 45, "min": 20,
                 "fact_prompt": "detailed", "fact_edge_prompt": "event"},
    "contracts": {"input_mode": "concat_file", "edge_mode": "clustered_fast", "max": 45, "min": 20,
                  "fact_prompt": "detailed", "fact_edge_prompt": "event"},
}


def qid(sample):
    return sample.get("id", sample.get("_id"))


def step_prepare(args, out: Path) -> Path:
    if args.dataset == "contracts":
        corpus = [{"title": "", "text": (Path(args.inputs) / "contract.txt").read_text(encoding="utf-8")}]
    elif args.dataset == "gmedical":
        corpus = json.load(open(Path(args.inputs) / "documents.json", encoding="utf-8"))
    else:
        corpus = json.load(open(DATA_ROOT / args.dataset / f"{args.dataset}_corpus.json", encoding="utf-8"))
    path = out / "data" / "corpus.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, doc in enumerate(corpus):
            f.write(json.dumps({"id": f"p{i:06d}", "title": doc["title"], "paragraph": doc["text"]}, ensure_ascii=False) + "\n")
    print(f"[prepare] {path}: {len(corpus):,} documents", flush=True)
    return path


def step_construct(args, out: Path):
    corpus_file = out / "data" / "corpus.jsonl"
    if not corpus_file.exists():
        corpus_file = step_prepare(args, out)
    c = DATASET_CONSTRUCTION[args.dataset]
    cmd = [sys.executable, "-m", "nkg.construct", "--corpus-file", str(corpus_file), "--run-dir", str(out),
           "--input-mode", c["input_mode"], "--edge-mode", c["edge_mode"],
           "--clustered-max-facts-per-prompt", str(c["max"]), "--fast-cluster-min-size", str(c["min"]),
           "--fact-prompt", c["fact_prompt"], "--fact-edge-prompt", c["fact_edge_prompt"],
           "--model", args.model, "--api-base", args.base_url,
           "--max-workers", str(args.concurrency), "--max-sub-workers", str(args.concurrency)]
    # the API key reaches the child through DEEPSEEK_API_KEY (never on the command line)
    print("[construct] " + " ".join(cmd), flush=True)
    t0 = time.time()
    subprocess.run(cmd, cwd=REPO, check=True)
    if not (out / "construction_results.json").exists():
        sys.exit("[construct] construction_results.json not written (interrupted?); rerun to resume from the checkpoint")
    print(f"[construct] done in {time.time() - t0:.0f}s", flush=True)


_RETRIEVERS = {}


def load_questions(args) -> list:
    if args.dataset in ("contracts", "gmedical"):
        return json.load(open(Path(args.inputs) / "questions.json", encoding="utf-8"))
    return json.load(open(DATA_ROOT / args.dataset / f"{args.dataset}.json", encoding="utf-8"))


def step_retrieve(args, out: Path, view: str):
    from nkg.retrieval.ppr_engine import PPRRetriever
    from sentence_transformers import CrossEncoder, SentenceTransformer

    kg = out / "kg.graphml"
    if not kg.exists():
        sys.exit(f"[retrieve] {kg} missing: run --step construct first")
    samples = load_questions(args)
    if args.limit_q:
        samples = samples[: args.limit_q]
    mode, ctx_name = VIEWS[view]
    report_name = f"retrieval_report_{view}.json"
    if args.tag:  # a variant (e.g. hyb_p5): its own contexts / report next to the base ones
        ctx_name = ctx_name.replace(".json", f"_{args.tag}.json")
        report_name = f"retrieval_report_{view}_{args.tag}.json"
    overrides = json.loads(args.ppr_args) if args.ppr_args else {}
    if args.ppr_passages:
        overrides["ppr_passages"] = args.ppr_passages

    path = out / ctx_name
    done = {}
    if path.exists():
        done = {r["id"]: r for r in json.load(open(path, encoding="utf-8")) if r["status"] == "ok"}
    todo = [s for s in samples if qid(s) not in done]

    key = str(kg)
    if key not in _RETRIEVERS:  # one embedding pass per process: --view full,hypergraph,discourse reuses it
        slug = args.retrieval_model.replace("/", "__")
        _RETRIEVERS[key] = PPRRetriever(
            SentenceTransformer(args.retrieval_model, trust_remote_code=True), CrossEncoder(args.cross_encoder_model),
            graph_path=str(kg), drop_empty_entities=True, model_name=args.retrieval_model,
            embedding_cache_path=str(out / f"kg.{slug}.dropempty.retriever_emb.npz"),
            chunk_cache_path=str(out / f"kg.{slug}.chunk_emb.npz"))
    retriever = _RETRIEVERS[key]
    retrieval_args = {**retriever.ppr_args, **overrides, "mode": mode}
    print(f"[retrieve:{view}] {len(samples)} questions ({len(done)} already done), args={retrieval_args}", flush=True)
    tok = retriever.tokenizer(retrieval_args["budget_tokenizer"])  # context tokens counted with the budget's tokenizer

    def one(s):
        t0 = time.time()
        try:
            ctx, paths = retriever.retrieve(query=s["question"], mode=mode, return_raw_paths=True, **overrides)
            # distinct edges of the final paths by type (an entity hop = fact_entity + entity_fact = 2 edges)
            seen = {}
            for p in paths:
                for u, et, _, v in p.history:
                    seen.setdefault(frozenset((u, v)), et)
            return {"id": qid(s), "question": s["question"], "context_text": ctx, "context_chars": len(ctx),
                    "status": "ok", "error": None, "seconds": round(time.time() - t0, 2),
                    "edges_discourse": sum(et == "fact_fact" for et in seen.values()),
                    "edges_entity": sum(et in ("fact_entity", "entity_fact") for et in seen.values()),
                    "context_tokens": len(tok(ctx)["input_ids"])}
        except Exception as exc:  # noqa: BLE001 - recorded; the run exits non-zero below
            return {"id": qid(s), "question": s["question"], "context_text": "", "context_chars": 0,
                    "status": "error", "error": repr(exc), "seconds": round(time.time() - t0, 2)}

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(one, s) for s in todo]
        for i, f in enumerate(as_completed(futures), 1):
            r = f.result()
            done[r["id"]] = r
            if i % 50 == 0 or i == len(todo):
                print(f"[retrieve:{view}] {i}/{len(todo)}  errors={sum(x['status'] != 'ok' for x in done.values())}",
                      flush=True)
    rows = [done[qid(s)] for s in samples if qid(s) in done]
    json.dump(rows, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    report = {"dataset": args.dataset, "view": view, "tag": args.tag, "questions": len(rows),
              "retriever": "ppr", "retrieval_args": retrieval_args,
              "errors": sum(r["status"] != "ok" for r in rows),
              "empty_contexts": sum(not r["context_text"].strip() for r in rows),
              "mean_context_chars": round(sum(r["context_chars"] for r in rows) / max(1, len(rows))),
              **path_stats(rows), "seconds_this_run": round(time.time() - t0, 1),
              "dropped_empty_entities": retriever.dropped_empty_entities}
    json.dump(report, open(out / report_name, "w"), indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "retrieval_args"}, indent=2), flush=True)
    if report["errors"]:
        sys.exit(f"[retrieve:{view}] {report['errors']} questions failed (see {path.name}); rerun to retry only those")


def path_stats(rows) -> dict:
    """Mean context tokens (budget tokenizer) and the mean per-question share of dependency (fact->fact) vs
    participation (fact<->entity) edges among the distinct path edges, and of hops (an entity hop = 2 edges = 1 hop)."""
    rows = [r for r in rows if "edges_discourse" in r]
    if not rows:
        return {}
    e, h = [], []
    for r in rows:
        ff, en = r["edges_discourse"], r["edges_entity"]
        if ff + en:
            e.append(ff / (ff + en))
            h.append(ff / (ff + en / 2))
    pct = lambda x: round(100 * sum(x) / len(x), 1) if x else None  # noqa: E731
    return {"mean_context_tokens": round(sum(r["context_tokens"] for r in rows) / len(rows)),
            "max_context_tokens": max(r["context_tokens"] for r in rows),
            "pct_edges_discourse": pct(e), "pct_edges_entity": None if not e else round(100 - pct(e), 1),
            "pct_hops_discourse": pct(h), "pct_hops_entity": None if not h else round(100 - pct(h), 1),
            "mean_edges_per_question": round(sum(r["edges_discourse"] + r["edges_entity"] for r in rows) / len(rows), 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(DATASET_CONSTRUCTION))
    ap.add_argument("--inputs", default=None, help="gmedical: inputs dir (default evaluation_data/gmedical/inputs); "
                    "contracts: evaluation_data/contracts/long_qa/<contract>")
    ap.add_argument("--out-dir", default=None, help="default: evaluation_scripts/unh/outputs/<dataset>")
    ap.add_argument("--step", default="all", choices=["prepare", "construct", "retrieve", "all"])
    ap.add_argument("--view", default="full", help="retrieve: full | hypergraph | discourse, or a comma list")
    ap.add_argument("--ppr-passages", type=int, default=0,
                    help="retrieve: append the top-N chunks by PPR score (UDH+5p: 5, with --tag hyb_p5)")
    ap.add_argument("--tag", default="", help="retrieve: variant tag -> contexts[_<view>]_<tag>.json")
    ap.add_argument("--ppr-args", default="", help="retrieve: JSON overrides of DEFAULT_PPR_ARGS (default: none)")
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", "deepseek-flash"))
    ap.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"))
    ap.add_argument("--concurrency", type=int, default=int(os.environ.get("UNH_CONCURRENCY", "2300")),
                    help="construction --max-workers / --max-sub-workers (default 2300; env UNH_CONCURRENCY). The HTTP "
                         "pool is sized by LLM_MAX_CONNECTIONS (nkg.utils.config, default 2500)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("UNH_RETRIEVAL_WORKERS", "16")),
                    help="questions retrieved in parallel")
    ap.add_argument("--retrieval-model", default=os.environ.get("EMBEDDING_MODEL", "voyageai/voyage-4-nano"))
    ap.add_argument("--cross-encoder-model", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    ap.add_argument("--limit-q", type=int, default=0, help="first N questions only (quick tests)")
    args = ap.parse_args()
    views = args.view.split(",")
    if any(v not in VIEWS for v in views):
        ap.error(f"--view: each of {args.view!r} must be one of {list(VIEWS)}")
    if args.dataset == "gmedical":
        args.inputs = args.inputs or str(DATA_ROOT / "gmedical" / "inputs")
    if args.dataset == "contracts" and not (args.inputs and args.out_dir):
        sys.exit("--dataset contracts needs --inputs <contract inputs dir> and --out-dir")
    if args.step in ("construct", "all") and not os.environ.get("DEEPSEEK_API_KEY"):
        sys.exit("export DEEPSEEK_API_KEY first (source evaluation_scripts/common/env.sh)")

    out = Path(args.out_dir).resolve() if args.out_dir else HERE / "outputs" / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    if args.step in ("prepare", "all"):
        step_prepare(args, out)
    if args.step in ("construct", "all"):
        step_construct(args, out)
    if args.step in ("retrieve", "all"):
        failed = []
        for view in views:
            try:
                step_retrieve(args, out, view)
            except SystemExit as exc:
                if exc.code not in (None, 0):
                    print(exc.code, flush=True)
                    failed.append(view)
        if failed:
            sys.exit(f"[retrieve] views with failures: {failed} (rerun resumes)")


if __name__ == "__main__":
    main()
