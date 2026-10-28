#!/usr/bin/env python3
"""Contract QA result tables from contracts_results/<set>/<system>/summary.json (common/contracts_score.py).

    python evaluation_scripts/common/collect_contracts.py [--set long_qa]

Written to contracts_results/<set>/tables.md (and printed) and contracts_results/<set>/qa.csv:
  QA            rows = systems; CAT1..CAT5 and All pooled over every question of the set, All excl. CAT5, Macro (mean
                of the per-contract scores), mean reader-context tokens (cl100k). Scores: mean judge score x 100
                (1.0 / 0.5 / 0.0 per question).
  Per contract  the same table for each contract.
  Diagnostics   empty contexts, abstentions on answerable questions, unparsed judge replies, reader truncation.
"""
import argparse
import csv
import json
from pathlib import Path

RESULTS_ROOT = Path(__file__).resolve().parents[1] / "contracts_results"
DATA_ROOT = Path(__file__).resolve().parents[2] / "evaluation_data" / "contracts"
CATS = [("CAT1", "CAT1_ATOMIC_LOOKUP"), ("CAT2", "CAT2_RELATIONAL_LOOKUP"), ("CAT3", "CAT3_CONDITIONAL_REASONING"),
        ("CAT4", "CAT4_MULTIHOP_REASONING"), ("CAT5", "CAT5_UNANSWERABLE")]
ORDER = ["hipporag2", "memgraphrag", "hygrag", "unh", "unh_hyb_p5", "ms_graphrag", "hypergraphrag", "logicrag",
         "unh_hypergraph", "unh_discourse"]
NAMES = {"unh": "UDH", "unh_hyb_p5": "UDH+5p", "unh_hypergraph": "UDH-H", "unh_discourse": "UDH-D"}


def fmt(x):
    return "-" if x is None else (f"{x:.1f}" if isinstance(x, float) else str(x))


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows]
    return "\n".join(out)


def exact(rows: list) -> dict:
    """Scores at full precision from the per-question rows (summary.json stores them rounded to 2 decimals)."""
    m = lambda xs: 100 * sum(xs) / len(xs) if xs else None  # noqa: E731
    return {"score_all": m([r["score"] for r in rows]),
            "score_excl_cat5": m([r["score"] for r in rows if r["category"] != "CAT5_UNANSWERABLE"]),
            "by_category": {k: {"score": m([r["score"] for r in rows if r["category"] == k])} for _, k in CATS},
            "mean_context_tokens": sum(r["context_tokens"] for r in rows) / len(rows), "n": len(rows)}


def collect(set_name: str):
    root = RESULTS_ROOT / set_name
    summaries = {}
    for p in sorted(root.glob("*/summary.json")):
        s = json.load(open(p))
        per = {c.name: json.load(open(c / "scores_qa.json"))["per_question"] for c in sorted(p.parent.iterdir())
               if (c / "scores_qa.json").exists()}
        s["per_contract"] = {c: {"qa": exact(r)} for c, r in per.items()}
        s["qa"]["pooled"] = {**s["qa"]["pooled"], **exact([x for r in per.values() for x in r])}
        s["qa"]["macro"]["score_all"] = sum(v["qa"]["score_all"] for v in s["per_contract"].values()) / len(per)
        summaries[s["system"]] = s
    if not summaries:
        return None
    systems = [s for s in ORDER if s in summaries] + sorted(set(summaries) - set(ORDER))
    contracts = json.load(open(DATA_ROOT / set_name / "contracts.json"))["contracts"]

    header = ["System"] + [c for c, _ in CATS] + ["All", "All excl. CAT5", "Macro", "Ctx tokens", "n"]
    rows = []
    for sysname in systems:
        s = summaries[sysname]
        p, m = s["qa"]["pooled"], s["qa"]["macro"]
        name = NAMES.get(sysname, sysname) + ("" if s.get("complete_set", True) else " (partial)")
        rows.append([name] + [p["by_category"][k]["score"] for _, k in CATS]
                    + [p["score_all"], p["score_excl_cat5"], m["score_all"], p["mean_context_tokens"], p["n"]])

    parts = [f"# Contract QA: {set_name}\n",
             "Scores in percent: mean judge score (1.0 / 0.5 / 0.0) per question; All = over every question of the set, "
             "Macro = mean of the per-contract scores. Ctx tokens: mean reader-context length (cl100k).\n",
             "## QA\n", md_table(header, rows), "\n## QA per contract\n"]
    for c in contracts:
        crows = []
        for sysname in systems:
            per = summaries[sysname]["per_contract"].get(c)
            if per:
                q = per["qa"]
                crows.append([NAMES.get(sysname, sysname)] + [q["by_category"][k]["score"] for _, k in CATS]
                             + [q["score_all"], q["score_excl_cat5"], q["mean_context_tokens"], q["n"]])
        if crows:
            parts += [f"### {c}\n", md_table(["System"] + [c_ for c_, _ in CATS] + ["All", "All excl. CAT5",
                                                                                   "Ctx tokens", "n"], crows), ""]
    diag = []
    for sysname in systems:
        p = summaries[sysname]["qa"]["pooled"]
        diag.append([NAMES.get(sysname, sysname), p["empty_contexts"], p["abstained_on_answerable"],
                     p["judge_unparsed"], p["reader_hit_max_tokens"], p["reader_no_answer_tag"]])
    parts += ["## Diagnostics\n", md_table(["System", "Empty contexts", "Abstained on CAT1-4", "Judge unparsed",
                                           "Reader hit max_tokens", "Reader without 'Answer:'"], diag)]
    text = "\n".join(parts) + "\n"
    (root / "tables.md").write_text(text)
    with open(root / "qa.csv", "w", newline="") as fh:
        csv.writer(fh).writerows([header] + rows)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="long_qa")
    args = ap.parse_args()
    text = collect(args.set)
    if text:
        print(text)
        print(f"(written to {RESULTS_ROOT / args.set / 'tables.md'})\n")


if __name__ == "__main__":
    main()
