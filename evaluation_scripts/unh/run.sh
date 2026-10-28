#!/usr/bin/env bash
# UDH on HotpotQA or MuSiQue (1,000 questions; the whole corpus as one index, one passage = one chunk):
#   construct  unh/main.py --step construct (DeepSeek; resumable)
#   retrieve   PPR retriever, the paper's setting, every view (UDH, UDH-H, UDH-D) and UDH+5p
#   answer     the shared answer step (common/answer.py) and common/score.py with the LLM judge, for each of the four
# Usage: bash evaluation_scripts/unh/run.sh hotpotqa|musique
#   UNH_CONCURRENCY        construction --max-workers / --max-sub-workers (default 2300)
#   UNH_RETRIEVAL_WORKERS  questions retrieved in parallel (default 16)
#   UNH_GPU                GPU for voyage-4-nano / the cross-encoder (default 1)
set -euo pipefail
DATASET="${1:?usage: run.sh hotpotqa|musique}"
source "$(dirname "$0")/../common/env.sh"
cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="${UNH_GPU:-1}"
OUT="$EVAL_ROOT/unh/outputs/$DATASET"
MAIN=("$PYTHON" "$EVAL_ROOT/unh/main.py" --dataset "$DATASET" --out-dir "$OUT")

"${MAIN[@]}" --step prepare
"${MAIN[@]}" --step construct
"${MAIN[@]}" --step retrieve --view full,hypergraph,discourse
"${MAIN[@]}" --step retrieve --view full --ppr-passages 5 --tag hyb_p5

for sys in unh unh_hyb_p5 unh_hypergraph unh_discourse; do
  case "$sys" in
    unh) ctx="$OUT/contexts.json" ;;
    unh_hyb_p5) ctx="$OUT/contexts_hyb_p5.json" ;;
    unh_hypergraph) ctx="$OUT/contexts_hypergraph.json" ;;
    unh_discourse) ctx="$OUT/contexts_discourse.json" ;;
  esac
  "$PYTHON" "$EVAL_ROOT/common/answer.py" --dataset "$DATASET" --system unh --retrieval "$ctx" \
    --out "$OUT/answers_${sys}_$DATASET.json" --workers "$(scale_concurrency 16)"
  "$PYTHON" "$EVAL_ROOT/common/score.py" --dataset "$DATASET" --system unh \
    --predictions "$OUT/answers_${sys}_$DATASET.json" --out "$OUT/scores_${sys}_$DATASET.json" \
    ${JUDGE:+--llm-judge --judge-model "$LLM_MODEL" --judge-base-url "$LLM_BASE_URL"} --workers "$(scale_concurrency 16)"
done
