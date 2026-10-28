#!/usr/bin/env bash
# HippoRAG 2 on G-Medical (GraphRAG-Bench Medical, as MemGraphRAG evaluates it).
# Index + retrieval over the G-Medical inputs (evaluation_data/build_gmedical_inputs.py: the medical corpus as one
# document, 2,062 questions), read by this system's document-collection loader (--dataset docs, the same loader the
# Contract QA runner uses). Then the shared answer step (common/answer.py --dataset gmedical: LinearRAG's zero-shot
# answer prompt) and common/score_gmedical.py (LLM-Acc, overall and per GraphRAG-Bench question type).
# Usage: bash evaluation_scripts/hipporag2/run_gmedical.sh
#   GMEDICAL_INPUTS  inputs dir (default evaluation_data/gmedical/inputs)
#   GMEDICAL_TAG     output name (default gmedical) -> outputs/<tag>_docs (HippoRAG appends the dataset label)
#   CONCURRENCY_MULTIPLIER  multiplies the LLM concurrency settings (common/env.sh)
set -euo pipefail
source "$(dirname "$0")/../common/env.sh"
cd "$EVAL_ROOT/hipporag2"
export OPENAI_API_KEY="$DEEPSEEK_API_KEY"      # HippoRAG's OpenAI client reads OPENAI_API_KEY
export GMEDICAL_INPUTS="${GMEDICAL_INPUTS:-$REPO_ROOT/evaluation_data/gmedical/inputs}"
export DOCS_INPUTS="$GMEDICAL_INPUTS"      # the document-collection loader reads this inputs dir
TAG="${GMEDICAL_TAG:-gmedical}"
PYTHON="${PYTHON:-python}"
SAVE="outputs/$TAG"; OUT="outputs/${TAG}_docs"
"$PYTHON" main.py --dataset docs --docs_inputs "$DOCS_INPUTS" --save_dir "$SAVE" \
  --llm_base_url "$LLM_BASE_URL" --llm_name "$LLM_MODEL" \
  --embedding_provider vllm --embedding_name "$EMBEDDING_MODEL" \
  --embedding_base_url "$EMBEDDING_BASE_URL/embeddings" \
  --openie_max_workers "$(scale_concurrency 8)" \
  --force_index_from_scratch "${HIPPORAG_FORCE_INDEX:-false}"
"$PYTHON" "$EVAL_ROOT/common/answer.py" --dataset gmedical --system hipporag2 --retrieval "$OUT/predictions.json" --workers "$(scale_concurrency 16)"
"$PYTHON" "$EVAL_ROOT/common/score_gmedical.py" --system hipporag2 --predictions "$OUT/answers_hipporag2_gmedical.json" --workers "$(scale_concurrency 16)"
