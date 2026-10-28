#!/usr/bin/env bash
# LogicRAG on G-Medical (GraphRAG-Bench Medical, as MemGraphRAG evaluates it).
# Index + retrieval over the G-Medical inputs (evaluation_data/build_gmedical_inputs.py: the medical corpus as one
# document, 2,062 questions), read by this system's document-collection loader (--dataset docs, the same loader the
# Contract QA runner uses). Then the shared answer step (common/answer.py --dataset gmedical: LinearRAG's zero-shot
# answer prompt) and common/score_gmedical.py (LLM-Acc, overall and per GraphRAG-Bench question type).
# Usage: bash evaluation_scripts/logicrag/run_gmedical.sh [question limit, for a quick test]
#   GMEDICAL_INPUTS  inputs dir (default evaluation_data/gmedical/inputs)
#   GMEDICAL_TAG     output name (default gmedical)
#   CONCURRENCY_MULTIPLIER  multiplies the LLM concurrency settings (common/env.sh)
set -euo pipefail
LIMIT="${1:-0}"
source "$(dirname "$0")/../common/env.sh"
cd "$EVAL_ROOT/logicrag"
export OPENAI_API_KEY="$DEEPSEEK_API_KEY" OPENAI_BASE_URL="$LLM_BASE_URL"
export LOGICRAG_LLM_MODEL="$LLM_MODEL" LOGICRAG_EMBEDDING_MODEL="$EMBEDDING_MODEL"
export LOGICRAG_CALLS_PER_MINUTE="${LOGICRAG_CALLS_PER_MINUTE:-600}"   # upstream default 20/min is an OpenAI-tier limit
export GMEDICAL_INPUTS="${GMEDICAL_INPUTS:-$REPO_ROOT/evaluation_data/gmedical/inputs}"
export DOCS_INPUTS="$GMEDICAL_INPUTS"      # the document-collection loader reads this inputs dir
TAG="${GMEDICAL_TAG:-gmedical}"
# own result / embedding-cache dirs: LogicRAG keys its embedding cache only on the corpus size
export LOGICRAG_RESULT_DIR="$EVAL_ROOT/logicrag/result/$TAG" LOGICRAG_CACHE_DIR="$EVAL_ROOT/logicrag/cache/$TAG"
PYTHON="${PYTHON:-python}"
OUT="$LOGICRAG_RESULT_DIR/logicrag_gmedical.json"
"$PYTHON" run.py --dataset "$DOCS_INPUTS/questions.json" --corpus "$DOCS_INPUTS/chunks_logicrag.json" \
  --limit "$LIMIT" --max-rounds 3 --top-k 5 --output logicrag_gmedical.json
"$PYTHON" "$EVAL_ROOT/common/answer.py" --dataset gmedical --system logicrag --retrieval "$OUT" --workers "$(scale_concurrency 16)"
"$PYTHON" "$EVAL_ROOT/common/score_gmedical.py" --system logicrag --predictions "$LOGICRAG_RESULT_DIR/answers_logicrag_gmedical.json" --workers "$(scale_concurrency 16)"
