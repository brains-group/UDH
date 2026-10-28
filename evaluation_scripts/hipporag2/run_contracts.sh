#!/usr/bin/env bash
# HippoRAG 2 on Contract QA (the five long-form contracts): one index per contract over the 1200/100-token chunks
# HyGRAG's chunker cuts from it (HippoRAG 2 has no chunker), top-5 passages retrieved for every query, then the shared answer step + QA judge (common/contracts_score.py).
# The contract inputs dir has the document-collection layout, read by main.py's loader (--dataset docs).
# Usage: bash evaluation_scripts/hipporag2/run_contracts.sh [long_qa]     (CONTRACTS="a b" for some only)
#   HIPPORAG_OPENIE_WORKERS  OpenIE workers per contract (default 8, upstream default)
set -euo pipefail
SET="${1:-long_qa}"
source "$(dirname "$0")/../common/env_contracts.sh"
source "$EVAL_ROOT/common/contracts_lib.sh"
cd "$EVAL_ROOT/hipporag2"
export OPENAI_API_KEY="$DEEPSEEK_API_KEY"
run_one() {
  "$PYTHON" main.py --dataset docs --docs_inputs "$CONTRACTS_DATA/$SET/$1" --save_dir "outputs/contracts_$SET/$1" \
    --llm_base_url "$LLM_BASE_URL" --llm_name "$LLM_MODEL" \
    --embedding_provider vllm --embedding_name "$EMBEDDING_MODEL" --embedding_base_url "$EMBEDDING_BASE_URL/embeddings" \
    --openie_max_workers "${HIPPORAG_OPENIE_WORKERS:-8}" --force_index_from_scratch "${HIPPORAG_FORCE_INDEX:-false}"
}
for_each_contract hipporag2 "$SET" run_one
score_contracts hipporag2 "$SET"
