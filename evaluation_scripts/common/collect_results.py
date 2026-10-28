#!/usr/bin/env python3
"""Collect every scores_<system>_<dataset>.json (common/score.py, common/score_gmedical.py) into result tables.

    python evaluation_scripts/common/collect_results.py      # markdown tables + results_table.csv

HotpotQA / MuSiQue: EM, F1, Contain-Acc, LLM-Acc (common/score.py). G-Medical: LLM-Acc overall and per question type,
plus the macro mean of the four types (common/score_gmedical.py). Contract QA: common/collect_contracts.py.
Context size (Ctx, mean reader-context tokens): common/verify_results.py.

Each scores file = {"dataset", "system", "num_questions", "metrics_percent": {...}, "judge_model", "per_example": [...]};
per_example holds every question's prediction and per-question scores, answers_<system>_<dataset>.json next to it the
context the reader saw and its raw reply.
"""
import argparse
import csv
import json
import re
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parents[1]
PATTERN = re.compile(r"^scores_(\w+?)_(hotpotqa|musique|gmedical)\.json$")
ORDER = ["hipporag2", "memgraphrag", "hygrag", "unh", "unh_hyb_p5", "ms_graphrag", "hypergraphrag", "logicrag",
         "unh_hypergraph", "unh_discourse"]
NAMES = {"unh": "UDH", "unh_hyb_p5": "UDH+5p", "unh_hypergraph": "UDH-H", "unh_discourse": "UDH-D"}
GMED_LEVELS = ["Fact Retrieval", "Complex Reasoning", "Contextual Summarize", "Creative Generation"]
GMED = ["LLM-Acc", "LLM-Acc Fact Retrieval", "LLM-Acc Complex Reasoning", "LLM-Acc Contextual Summarize",
        "LLM-Acc Creative Generation", "LLM-Acc macro (mean of levels)"]


def exact_metrics(data: dict) -> dict:
    """metrics_percent recomputed from per_example at full precision (the stored values are rounded to 2 decimals,
    so rounding them again to 1 decimal can differ from the exact mean)."""
    pe = data["per_example"]
    out = {}
    for k in data["metrics_percent"]:
        if k in pe[0]:
            out[k] = 100 * sum(float(r[k]) for r in pe) / len(pe)
        elif k.startswith("LLM-Acc ") and k[8:] in GMED_LEVELS:
            sel = [float(r["LLM-Acc"]) for r in pe if r.get("level") == k[8:]]
            out[k] = 100 * sum(sel) / len(sel)
    if "LLM-Acc macro (mean of levels)" in data["metrics_percent"]:
        out["LLM-Acc macro (mean of levels)"] = sum(out[f"LLM-Acc {lv}"] for lv in GMED_LEVELS) / len(GMED_LEVELS)
    return out


def table(header, rows):
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in rows:
        print("| " + " | ".join(r) + " |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(EVAL_ROOT), help="directory to search (default: evaluation_scripts/)")
    ap.add_argument("--csv", default=None, help="default: <root>/results_table.csv")
    args = ap.parse_args()
    root = Path(args.root)

    rows = []
    for path in sorted(root.rglob("scores_*.json")):
        m = PATTERN.match(path.name)
        if not m:
            continue
        data = json.load(open(path, encoding="utf-8"))
        rows.append({"system": m.group(1), "dataset": m.group(2), "n": data["num_questions"],
                     "judge": data.get("judge_model"), **exact_metrics(data), "file": str(path.relative_to(root))})
    if not rows:
        raise SystemExit(f"no scores_*.json under {root}")
    by = {(r["system"], r["dataset"]): r for r in rows}
    systems = [s for s in ORDER if any(k[0] == s for k in by)] + sorted({r["system"] for r in rows} - set(ORDER))

    def cell(system, ds, metric):
        r = by.get((system, ds))
        return f"{r[metric]:.1f}" if r and metric in r else "–"

    print("## HotpotQA and MuSiQue (1,000 questions each)\n")
    wiki = [s for s in systems if (s, "hotpotqa") in by or (s, "musique") in by]
    table(["Method"] + [f"{ds} {m}" for ds in ("HotpotQA", "MuSiQue") for m in ("EM", "F1", "LLM-Acc")],
          [[NAMES.get(s, s)] + [cell(s, ds, m) for ds in ("hotpotqa", "musique") for m in ("EM", "F1", "LLM-Acc")]
           for s in wiki])
    print("\n## G-Medical (2,062 questions; LLM-Acc)\n")
    gmed = [s for s in systems if (s, "gmedical") in by]
    table(["Method", "All", "Fact Retrieval", "Complex Reasoning", "Contextual Summarize", "Creative Generation", "Macro"],
          [[NAMES.get(s, s)] + [cell(s, "gmedical", m) for m in GMED] for s in gmed])

    out_csv = Path(args.csv) if args.csv else root / "results_table.csv"
    fields = ["system", "dataset", "n", "EM", "F1", "Contain-Acc", "LLM-Acc"] + GMED[1:] + ["judge", "file"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows([{k: round(v, 2) if isinstance(v, float) else v for k, v in r.items()} for r in rows])
    judges = sorted({r["judge"] for r in rows if r["judge"]})
    print(f"\nLLM judge: {', '.join(judges) or 'none'}   |   all rows: {out_csv.relative_to(root) if out_csv.is_relative_to(root) else out_csv}")


if __name__ == "__main__":
    main()
