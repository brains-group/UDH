#!/usr/bin/env python3
"""Re-derive every number of the paper's result tables from the per-question records.

    python evaluation_scripts/common/verify_results.py [--skip-tokens]

For each (dataset, system) it opens the scores file, recomputes the metrics as plain means over the per-question rows
(HotpotQA / MuSiQue / G-Medical: `per_example`; Contract QA: every contract's scores_qa.json), prints them next to the
file, and flags (!!) any difference from the stored summary. Ctx = mean reader-context tokens per question (cl100k):
the context the shared reader saw (answers_*.json `retrieved_docs`, joined with blank lines; Contract QA: the stored
`context_tokens`). Read-only: writes nothing.
"""
import argparse
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = ["hipporag2", "memgraphrag", "hygrag", "unh", "unh_hyb_p5", "ms_graphrag", "hypergraphrag", "logicrag",
           "unh_hypergraph", "unh_discourse"]
NAMES = {"unh": "UDH", "unh_hyb_p5": "UDH+5p", "unh_hypergraph": "UDH-H", "unh_discourse": "UDH-D"}
LEVELS = ["Fact Retrieval", "Complex Reasoning", "Contextual Summarize", "Creative Generation"]
CATS = ["CAT1_ATOMIC_LOOKUP", "CAT2_RELATIONAL_LOOKUP", "CAT3_CONDITIONAL_REASONING", "CAT4_MULTIHOP_REASONING",
        "CAT5_UNANSWERABLE"]


def out_dir(system: str, ds: str) -> Path:
    if system.startswith("unh"):
        return ROOT / "unh/outputs" / ds
    if system == "logicrag":
        return ROOT / "logicrag/result" / ("" if ds in ("hotpotqa", "musique") else ds)
    if system == "hipporag2" and ds == "gmedical":
        return ROOT / "hipporag2/outputs/gmedical_docs"
    return ROOT / system / "outputs" / ds


def load(path: Path):
    if not path.exists() and Path(str(path) + ".gz").exists():
        with gzip.open(str(path) + ".gz", "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.load(open(path, encoding="utf-8"))


_enc = None


def ctx_tokens(answers_path: Path):
    global _enc
    if _enc is None:
        import tiktoken
        _enc = tiktoken.get_encoding("cl100k_base")
    rows = load(answers_path)
    return sum(len(_enc.encode("\n\n".join(map(str, r.get("retrieved_docs") or [])), disallowed_special=()))
               for r in rows) / max(len(rows), 1)


def mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return 100 * sum(xs) / len(xs) if xs else None


def check(label, stored, recomputed):
    ok = stored is None or recomputed is None or abs(stored - recomputed) < 0.06
    return f"{label} {recomputed:.1f}" + ("" if ok else f" (!! stored {stored})")


def wiki(ds, tokens):
    print(f"\n## {ds} (scores_<system>_{ds}.json)")
    for s in SYSTEMS:
        d = out_dir(s, ds)
        f = d / f"scores_{s}_{ds}.json"
        if not f.exists():
            continue
        data = load(f)
        pe, st = data["per_example"], data["metrics_percent"]
        parts = [check(m, st.get(m), mean(r[m] for r in pe)) for m in ("EM", "F1", "LLM-Acc")]
        if tokens:
            parts.append(f"Ctx {ctx_tokens(d / f'answers_{s}_{ds}.json'):,.0f}")
        print(f"  {NAMES.get(s, s):14s} n={len(pe):5d}  " + "  ".join(parts) + f"   [{f.relative_to(ROOT)}]")


def gmedical(tokens):
    print("\n## gmedical (scores_<system>_gmedical.json; Macro = mean of the four question types)")
    for s in SYSTEMS:
        d = out_dir(s, "gmedical")
        f = d / f"scores_{s}_gmedical.json"
        if not f.exists():
            continue
        data = load(f)
        pe, st = data["per_example"], data["metrics_percent"]
        per_level = {lv: mean(r["LLM-Acc"] for r in pe if r["level"] == lv) for lv in LEVELS}
        parts = [check("All", st["LLM-Acc"], mean(r["LLM-Acc"] for r in pe))]
        parts += [check(lv.split()[0], st.get(f"LLM-Acc {lv}"), per_level[lv]) for lv in LEVELS]
        parts.append(check("Macro", st.get("LLM-Acc macro (mean of levels)"), sum(per_level.values()) / 4))
        if tokens:
            parts.append(f"Ctx {ctx_tokens(d / f'answers_{s}_gmedical.json'):,.0f}")
        print(f"  {NAMES.get(s, s):14s} n={len(pe):5d}  " + "  ".join(parts) + f"   [{f.relative_to(ROOT)}]")


def contracts(set_name="long_qa"):
    root = ROOT / "contracts_results" / set_name
    print(f"\n## Contract QA (contracts_results/{set_name}/<system>/<contract>/scores_qa.json; "
          "score = mean judge score; Macro = mean of the per-contract scores)")
    for s in SYSTEMS:
        summ = root / s / "summary.json"
        if not summ.exists():
            continue
        st = load(summ)["qa"]
        rows, per_contract = [], []
        for f in sorted((root / s).glob("*/scores_qa.json")):
            r = load(f)["per_question"]
            rows += r
            per_contract.append(mean(x["score"] for x in r))
        parts = [check("All", st["pooled"]["score_all"], mean(x["score"] for x in rows))]
        parts += [check(c[:4], st["pooled"]["by_category"][c]["score"],
                        mean(x["score"] for x in rows if x["category"] == c)) for c in CATS]
        parts.append(check("Macro", st["macro"]["score_all"], sum(per_contract) / len(per_contract)))
        parts.append(f"Ctx {sum(x['context_tokens'] for x in rows) / len(rows):,.0f}")
        print(f"  {NAMES.get(s, s):14s} n={len(rows):5d}  " + "  ".join(parts) + f"   [{summ.relative_to(ROOT)}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-tokens", action="store_true", help="skip the context-token counts (faster)")
    args = ap.parse_args()
    for ds in ("hotpotqa", "musique"):
        wiki(ds, not args.skip_tokens)
    gmedical(not args.skip_tokens)
    contracts()


if __name__ == "__main__":
    main()
