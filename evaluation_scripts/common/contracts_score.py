#!/usr/bin/env python3
"""Contract QA: the shared answer step + the QA judge for one system over the contract set.

    python evaluation_scripts/common/contracts_score.py --system hipporag2 --set long_qa

Every system built one graph per contract and retrieved one context per question
(evaluation_data/contracts/<set>/<contract>/questions.json). This script reads each contract's retrieval file (the
system's own output, the same context the HotpotQA / MuSiQue answer step reads, see answer.load_contexts) and runs
  the shared answer step: answer.Reader with the contract prompt (prompts.build_contract_messages: written reasoning
    after "Thought:", final answer after "Answer:", abstain phrase "Not specified in this policy.")
  -> the QA judge (prompts.contract_qa_judge_messages: 1.0 / 0.5 / 0.0, written reasoning, "Score: X"). The judge sees
    the question, the gold answer and the answer, never the context.

Every call: same LLM (LLM_MODEL at LLM_BASE_URL, DeepSeek), max_tokens 4096, no temperature sent unless
LLM_TEMPERATURE, cached (reader_cache.sqlite / judge_cache.sqlite), usage recorded. A judge reply without a parseable
"Score:" is retried twice (never cached), then scored 0 and counted as judge_unparsed.

Outputs, evaluation_scripts/contracts_results/<set>/<system>/:
  <contract>/scores_qa.json  per question: id, category, question, gold, evidence, context (full), context_tokens
                             (cl100k), raw reply, answer, score, judge reply, usage; plus the contract summary
  summary.json               per contract and over the set (pooled = over all questions; macro = mean of the
                             per-contract scores), per category; scores in percent
"""
import argparse
import hashlib
import json
import os
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import tiktoken
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from answer import Reader, load_contexts, usage_record  # noqa: E402
from prompts import CONTRACT_ABSTAIN, contract_qa_judge_messages, parse_judge_score  # noqa: E402

EVAL_ROOT = HERE.parent
DATA_ROOT = EVAL_ROOT.parent / "evaluation_data" / "contracts"
RESULTS_ROOT = EVAL_ROOT / "contracts_results"
CATEGORIES = ["CAT1_ATOMIC_LOOKUP", "CAT2_RELATIONAL_LOOKUP", "CAT3_CONDITIONAL_REASONING",
              "CAT4_MULTIHOP_REASONING", "CAT5_UNANSWERABLE"]
ENC = tiktoken.get_encoding("cl100k_base")

# system -> (answer.load_contexts loader, retrieval file under EVAL_ROOT; {set}, {contract} filled in).
# These are the files each <system>/run_contracts.sh writes.
RETRIEVAL = {
    "hipporag2": ("hipporag2", "hipporag2/outputs/contracts_{set}/{contract}_quality/predictions.json"),
    "logicrag": ("logicrag", "logicrag/result/contracts_{set}/{contract}/logicrag.json"),
    "memgraphrag": ("memgraphrag", "memgraphrag/outputs/contracts_{set}/{contract}/qa_results.json"),
    "hypergraphrag": ("hypergraphrag", "hypergraphrag/outputs/contracts_{set}/{contract}/knowledge.json"),
    "ms_graphrag": ("ms_graphrag", "ms_graphrag/outputs/contracts_{set}/{contract}/local_search_contexts.json"),
    "hygrag": ("hygrag", "hygrag/outputs/contracts_{set}/{contract}/contexts.json"),
    "unh": ("unh", "unh/outputs/contracts_{set}/{contract}/contexts.json"),                       # UDH
    "unh_hyb_p5": ("unh", "unh/outputs/contracts_{set}/{contract}/contexts_hyb_p5.json"),         # UDH+5p
    "unh_hypergraph": ("unh", "unh/outputs/contracts_{set}/{contract}/contexts_hypergraph.json"), # UDH-H
    "unh_discourse": ("unh", "unh/outputs/contracts_{set}/{contract}/contexts_discourse.json"),   # UDH-D
}


class Judge:
    """Cached chat calls for the QA judge (same settings as the Reader)."""

    def __init__(self, model, base_url, api_key, max_tokens, temperature, cache_path: Path):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=600, max_retries=5)
        self.params = {"model": model, "max_tokens": max_tokens}
        if temperature is not None:
            self.params["temperature"] = temperature
        if os.environ.get("LLM_DISABLE_THINKING", "1") != "0":
            self.params["extra_body"] = {"thinking": {"type": "disabled"}}
        self.conn = sqlite3.connect(str(cache_path), check_same_thread=False, timeout=120)
        self.conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, response TEXT, usage TEXT)")
        self.conn.commit()
        self.lock = threading.Lock()
        self.api_calls_this_run = 0

    def __call__(self, messages, allowed):
        """-> (score or None, raw reply, usage summed over attempts, attempts)."""
        request = {**self.params, "messages": messages}
        key = hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        with self.lock:
            row = self.conn.execute("SELECT response, usage FROM cache WHERE key = ?", (key,)).fetchone()
        if row is not None:
            return parse_judge_score(row[0], allowed), row[0], json.loads(row[1]), 0
        total = {"prompt_tokens": 0, "completion_tokens": 0, "prompt_cache_hit_tokens": 0}
        raw, score = "", None
        for attempt in range(1, 4):
            response = self.client.chat.completions.create(**request)
            raw = response.choices[0].message.content or ""
            u = usage_record(response.usage)
            total = {k: total[k] + u[k] for k in total}
            with self.lock:
                self.api_calls_this_run += 1
            score = parse_judge_score(raw, allowed)
            if score is not None:
                with self.lock:
                    self.conn.execute("INSERT OR REPLACE INTO cache (key, response, usage) VALUES (?, ?, ?)",
                                      (key, raw, json.dumps(total)))
                    self.conn.commit()
                return score, raw, total, attempt
        return None, raw, total, 3


def ntok(text: str) -> int:
    return len(ENC.encode(text or "", disallowed_special=()))


def mean(xs):
    xs = list(xs)
    return round(100 * sum(xs) / len(xs), 2) if xs else None


def add_usage(rows, field):
    keys = ("prompt_tokens", "completion_tokens", "prompt_cache_hit_tokens")
    tot = {k: sum((r.get(field) or {}).get(k, 0) for r in rows) for k in keys}
    return {"calls": sum(1 for r in rows if r.get(field)), **tot}


def is_abstention(answer: str) -> bool:
    a = (answer or "").lower()
    return CONTRACT_ABSTAIN.lower().rstrip(".") in a or "not specified" in a or "does not contain enough information" in a


def qa_summary(rows):
    s = {"n": len(rows), "score_all": mean(r["score"] for r in rows),
         "score_excl_cat5": mean(r["score"] for r in rows if r["category"] != "CAT5_UNANSWERABLE"),
         "by_category": {c: {"score": mean(r["score"] for r in rows if r["category"] == c),
                             "n": sum(r["category"] == c for r in rows)} for c in CATEGORIES},
         "abstained_on_answerable": sum(r["abstained"] for r in rows if r["category"] != "CAT5_UNANSWERABLE"),
         "mean_context_tokens": round(sum(r["context_tokens"] for r in rows) / max(1, len(rows)), 1),
         "empty_contexts": sum(not r["context"].strip() for r in rows),
         "reader_empty": sum(not r["raw_response"] for r in rows),
         "reader_hit_max_tokens": sum(r["finish_reason"] == "length" for r in rows),
         "reader_no_answer_tag": sum("Answer:" not in r["raw_response"] for r in rows),
         "judge_unparsed": sum(r["judge_unparsed"] for r in rows),
         "reader_usage": add_usage(rows, "reader_usage"), "judge_usage": add_usage(rows, "judge_usage")}
    return s


def score_contract(args, contract, reader, judge, out_dir: Path):
    queries = json.load(open(DATA_ROOT / args.set / contract / "questions.json", encoding="utf-8"))
    loader, template = RETRIEVAL[args.system]
    path = EVAL_ROOT / template.format(set=args.set, contract=contract)
    if not path.exists():
        sys.exit(f"[{args.system}] {contract}: retrieval file missing: {path}")
    rows = load_contexts(loader, str(path), args.qa_top_k)
    if len(rows) != len(queries):
        sys.exit(f"[{args.system}] {contract}: {len(rows)} retrieved contexts for {len(queries)} queries in {path}")
    for i, ((q, _), s) in enumerate(zip(rows, queries)):
        if q.strip() != s["question"].strip():
            sys.exit(f"[{args.system}] {contract}: query mismatch at index {i}: {q!r} vs {s['question']!r}")
    contexts = ["\n\n".join(d for d in docs if d and d.strip()) for _, docs in rows]

    qa_idx = [i for i, s in enumerate(queries) if s["kind"] == "qa"]

    def answer_and_judge(i):
        s = queries[i]
        raw, pred, finish, r_usage = reader(s["question"], rows[i][1])
        score, j_raw, j_usage, _ = judge(contract_qa_judge_messages(s["question"], s["answer"], pred), (1.0, 0.5, 0.0))
        return {"id": s["id"], "category": s["category"], "question": s["question"], "gold": s["answer"],
                "evidence": s["evidence"], "context": contexts[i], "context_tokens": ntok(contexts[i]),
                "raw_response": raw, "answer": pred, "finish_reason": finish, "abstained": is_abstention(pred),
                "score": score or 0.0, "judge_unparsed": score is None, "judge_response": j_raw,
                "reader_usage": r_usage, "judge_usage": j_usage}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        qa_rows = list(tqdm(ex.map(answer_and_judge, qa_idx), total=len(qa_idx), desc=f"{args.system} {contract} QA"))

    out_dir.mkdir(parents=True, exist_ok=True)
    qa_sum = qa_summary(qa_rows)
    meta = {"system": args.system, "set": args.set, "contract": contract,
            "retrieval_file": str(path.relative_to(EVAL_ROOT))}
    json.dump({**meta, "summary": qa_sum, "per_question": qa_rows},
              open(out_dir / "scores_qa.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[{args.system}] {contract}: QA {qa_sum['score_all']} (excl CAT5 {qa_sum['score_excl_cat5']}), "
          f"ctx tokens {qa_sum['mean_context_tokens']}", flush=True)
    return qa_rows, qa_sum


def set_summary(per_contract, all_qa):
    def macro(get):
        vals = [get(c) for c in per_contract.values()]
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 2) if vals else None
    return {
        "qa": {"pooled": qa_summary(all_qa),
               "macro": {"score_all": macro(lambda c: c["qa"]["score_all"]),
                         "score_excl_cat5": macro(lambda c: c["qa"]["score_excl_cat5"]),
                         "by_category": {k: macro(lambda c, k=k: c["qa"]["by_category"][k]["score"]) for k in CATEGORIES},
                         "mean_context_tokens": macro(lambda c: c["qa"]["mean_context_tokens"])}},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", required=True, choices=list(RETRIEVAL))
    ap.add_argument("--set", default="long_qa", help="evaluation_data/contracts/<set>/ (long_qa: the five contracts)")
    ap.add_argument("--contracts", nargs="*", default=None, help="default: every contract of the set")
    ap.add_argument("--qa-top-k", type=int, default=5, help="docs passed on for hipporag2 / memgraphrag (as answer.py)")
    ap.add_argument("--max-tokens", type=int, default=4096, help="reader and judge (as G-Medical)")
    ap.add_argument("--temperature", default=os.environ.get("LLM_TEMPERATURE") or "none")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("CONTRACT_SCORE_WORKERS", "32")))
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL"))
    ap.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL"))
    ap.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    ap.add_argument("--results-root", default=str(RESULTS_ROOT))
    args = ap.parse_args()
    if not args.model or not args.base_url:
        sys.exit("LLM_MODEL / LLM_BASE_URL not set: source evaluation_scripts/common/env.sh")

    manifest = json.load(open(DATA_ROOT / args.set / "contracts.json"))
    contracts = args.contracts or manifest["contracts"]
    out_root = Path(args.results_root) / args.set / args.system
    out_root.mkdir(parents=True, exist_ok=True)
    temperature = None if str(args.temperature).lower() == "none" else float(args.temperature)
    api_key = os.environ.get(args.api_key_env) or "EMPTY"
    reader = Reader(args.model, args.base_url, api_key, args.max_tokens, temperature,
                    out_root / "reader_cache.sqlite", prompt="contracts")
    judge = Judge(args.model, args.base_url, api_key, args.max_tokens, temperature, out_root / "judge_cache.sqlite")

    t0 = time.time()
    per_contract, all_qa = {}, []
    for c in contracts:
        qa_rows, qa_sum = score_contract(args, c, reader, judge, out_root / c)
        timing = out_root / f"timing_{c}.json"
        per_contract[c] = {"qa": qa_sum,
                           "index_and_retrieval_seconds": json.load(open(timing)).get("index_and_retrieval_seconds")
                           if timing.exists() else None}
        all_qa += qa_rows
    summary = {"system": args.system, "set": args.set, "contracts": contracts,
               "complete_set": sorted(contracts) == sorted(manifest["contracts"]),
               "llm_model": args.model, "temperature": args.temperature,
               "max_tokens": args.max_tokens,
               "api_calls_this_run": {"reader": reader.api_calls_this_run, "judge": judge.api_calls_this_run},
               "seconds_this_run": round(time.time() - t0, 1),
               **set_summary(per_contract, all_qa), "per_contract": per_contract}
    json.dump(summary, open(out_root / "summary.json", "w"), indent=1)
    p, m = summary["qa"]["pooled"], summary["qa"]["macro"]
    print(json.dumps({"system": args.system, "set": args.set,
                      "qa_pooled": p["score_all"], "qa_pooled_excl_cat5": p["score_excl_cat5"], "qa_macro": m["score_all"],
                      "qa_by_category_pooled": {k: v["score"] for k, v in p["by_category"].items()},
                      "judge_unparsed": p["judge_unparsed"], "empty_contexts": p["empty_contexts"]}, indent=1))
    print(f"wrote {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
