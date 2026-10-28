#!/usr/bin/env bash
# HyperGraphRAG (upstream, unmodified) on Contract QA (the five long-form contracts): rag.insert(the whole
# contract as one document; HyperGraphRAG chunks it 1200/100 itself) -> hybrid-mode context for every question -> shared answer step + QA judge (common/contracts_score.py).
# The contract inputs dir has the document-collection layout, read by main.py's loader (--dataset docs).
# Usage: bash evaluation_scripts/hypergraphrag/run_contracts.sh [long_qa]     (CONTRACTS="a b" for some only)
set -euo pipefail
SET="${1:-long_qa}"
source "$(dirname "$0")/../common/env_contracts.sh"
source "$EVAL_ROOT/common/contracts_lib.sh"
cd "$EVAL_ROOT/hypergraphrag"
run_one() {  # upstream max_async values (index 32/32, retrieval 16/16, 32 queries in flight)
  "$PYTHON" main.py --dataset docs --docs-inputs "$CONTRACTS_DATA/$SET/$1" \
    --out-dir "$EVAL_ROOT/hypergraphrag/outputs/contracts_$SET/$1" --step all \
    --index-llm-max-async 32 --index-embedding-max-async 32 \
    --retrieve-llm-max-async 16 --retrieve-embedding-max-async 16 --retrieve-concurrency 32
}
for_each_contract hypergraphrag "$SET" run_one
score_contracts hypergraphrag "$SET"
