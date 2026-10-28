#!/usr/bin/env bash
# HyperGraphRAG on G-Medical (GraphRAG-Bench Medical, as MemGraphRAG evaluates it).
# Index + retrieval over the G-Medical inputs (evaluation_data/build_gmedical_inputs.py: the medical corpus as one
# document, 2,062 questions), read by this system's document-collection loader (--dataset docs, the same loader the
# Contract QA runner uses). Then the shared answer step (common/answer.py --dataset gmedical: LinearRAG's zero-shot
# answer prompt) and common/score_gmedical.py (LLM-Acc, overall and per GraphRAG-Bench question type).
# Usage: bash evaluation_scripts/hypergraphrag/run_gmedical.sh [question limit, for a quick test]
#   GMEDICAL_INPUTS  inputs dir (default evaluation_data/gmedical/inputs)
#   GMEDICAL_TAG     output name (default gmedical)
#   CONCURRENCY_MULTIPLIER  multiplies the LLM concurrency settings (common/env.sh)
set -euo pipefail
LIMIT="${1:-0}"
source "$(dirname "$0")/../common/env.sh"
cd "$EVAL_ROOT/hypergraphrag"
export GMEDICAL_INPUTS="${GMEDICAL_INPUTS:-$REPO_ROOT/evaluation_data/gmedical/inputs}"
export DOCS_INPUTS="$GMEDICAL_INPUTS"      # the document-collection loader reads this inputs dir
TAG="${GMEDICAL_TAG:-gmedical}"
PYTHON="${PYTHON:-python}"
OUT="$EVAL_ROOT/hypergraphrag/outputs/$TAG"
"$PYTHON" main.py --dataset docs --docs-inputs "$DOCS_INPUTS" --out-dir "$OUT" --step all --limit "$LIMIT" \
  --index-llm-max-async "$(scale_concurrency 32)" --index-embedding-max-async "$(scale_concurrency 32)" \
  --retrieve-llm-max-async "$(scale_concurrency 16)" --retrieve-embedding-max-async "$(scale_concurrency 16)" \
  --retrieve-concurrency "$(scale_concurrency 32)"
"$PYTHON" "$EVAL_ROOT/common/answer.py" --dataset gmedical --system hypergraphrag --retrieval "$OUT/knowledge.json" --workers "$(scale_concurrency 16)"
"$PYTHON" "$EVAL_ROOT/common/score_gmedical.py" --system hypergraphrag --predictions "$OUT/answers_hypergraphrag_gmedical.json" --workers "$(scale_concurrency 16)"
