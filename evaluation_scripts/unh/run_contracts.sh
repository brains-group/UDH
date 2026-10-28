#!/usr/bin/env bash
# UDH on Contract QA: one index per contract (the contract is one document, chunked 600/50 by UDH's chunker; unh/main.py's
# contract construction settings), then the PPR retriever (the paper's setting) for every question, in every view
# (UDH, UDH-H, UDH-D) and UDH+5p; then the shared answer step + QA judge (common/contracts_score.py) for each.
# Usage: bash evaluation_scripts/unh/run_contracts.sh [long_qa]     (CONTRACTS="a b" for some only)
#   UNH_CONCURRENCY   construction --max-workers / --max-sub-workers per contract (default 64)
#   UNH_GPU           GPU for voyage-4-nano / the cross-encoder (default 1)
set -euo pipefail
SET="${1:-long_qa}"
source "$(dirname "$0")/../common/env_contracts.sh"
source "$EVAL_ROOT/common/contracts_lib.sh"
cd "$REPO_ROOT"
export UNH_CONCURRENCY="${UNH_CONCURRENCY:-64}"
export CUDA_VISIBLE_DEVICES="${UNH_GPU:-1}"
run_one() {
  local MAIN=("$PYTHON" "$EVAL_ROOT/unh/main.py" --dataset contracts --inputs "$CONTRACTS_DATA/$SET/$1"
              --out-dir "$EVAL_ROOT/unh/outputs/contracts_$SET/$1")
  "${MAIN[@]}" --step prepare
  "${MAIN[@]}" --step construct
  "${MAIN[@]}" --step retrieve --view full,hypergraph,discourse
  "${MAIN[@]}" --step retrieve --view full --ppr-passages 5 --tag hyb_p5
}
for_each_contract unh "$SET" run_one
for sys in unh unh_hyb_p5 unh_hypergraph unh_discourse; do
  score_contracts "$sys" "$SET"
done
