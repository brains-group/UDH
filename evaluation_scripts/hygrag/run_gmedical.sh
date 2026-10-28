#!/usr/bin/env bash
# HyGRAG on G-Medical (GraphRAG-Bench Medical, as MemGraphRAG evaluates it).
# Index + retrieval over the G-Medical inputs (evaluation_data/build_gmedical_inputs.py: the medical corpus as one
# document, 2,062 questions), read by this system's document-collection loader (--dataset docs, the same loader the
# Contract QA runner uses). Then the shared answer step (common/answer.py --dataset gmedical: LinearRAG's zero-shot
# answer prompt) and common/score_gmedical.py (LLM-Acc, overall and per GraphRAG-Bench question type).
# Usage: bash evaluation_scripts/hygrag/run_gmedical.sh
#   GMEDICAL_INPUTS  inputs dir (default evaluation_data/gmedical/inputs)
#   GMEDICAL_TAG     output name (default gmedical)
#   CONCURRENCY_MULTIPLIER  multiplies the LLM concurrency settings (common/env.sh)
set -euo pipefail
source "$(dirname "$0")/../common/env.sh"
HERE="$EVAL_ROOT/hygrag"
cd "$HERE"
export GMEDICAL_INPUTS="${GMEDICAL_INPUTS:-$REPO_ROOT/evaluation_data/gmedical/inputs}"
export DOCS_INPUTS="$GMEDICAL_INPUTS"      # the document-collection loader reads this inputs dir
TAG="${GMEDICAL_TAG:-gmedical}"
HY_PY="${HYGRAG_PYTHON:-$HOME/miniconda3/envs/hygrag/bin/python}"
PYTHON="${PYTHON:-python}"
OUT="$HERE/outputs/$TAG"
export HYGRAG_CONCURRENCY="$(scale_concurrency "${HYGRAG_CONCURRENCY:-500}")"
echo "[hygrag] concurrency $HYGRAG_CONCURRENCY"
"$HY_PY" prepare_data.py --dataset docs --docs-inputs "$DOCS_INPUTS" --out-dir "data/$TAG"
# --allow-fallback: questions that hit a fallback stay flagged in contexts.json / run_report.json instead of stopping
# the run before the answer step (none of the HotpotQA / MuSiQue questions were flagged)
"$HY_PY" main.py --dataset docs --data-dir "data/$TAG" --out-dir "$OUT" --questions "$DOCS_INPUTS/questions.json" --allow-fallback
"$PYTHON" "$EVAL_ROOT/common/answer.py" --dataset gmedical --system hygrag --retrieval "$OUT/contexts.json" --workers "$(scale_concurrency 16)"
"$PYTHON" "$EVAL_ROOT/common/score_gmedical.py" --system hygrag --predictions "$OUT/answers_hygrag_gmedical.json" --workers "$(scale_concurrency 16)"
