#!/usr/bin/env bash
# UDH on G-Medical (GraphRAG-Bench Medical, as MemGraphRAG evaluates it):
#   construct  one index over $GMEDICAL_INPUTS/documents.json (the medical corpus as one document, chunked 600/50 by
#              UDH's chunker), unh/main.py's G-Medical construction settings (DeepSeek; resumable)
#   retrieve   PPR retriever, the paper's setting, every view (UDH, UDH-H, UDH-D) and UDH+5p
#   answer     the shared answer step (common/answer.py --dataset gmedical), then common/score_gmedical.py
# Usage: bash evaluation_scripts/unh/run_gmedical.sh
#   GMEDICAL_INPUTS        inputs dir (default evaluation_data/gmedical/inputs)
#   UNH_CONCURRENCY        construction --max-workers / --max-sub-workers (default 2300)
#   UNH_RETRIEVAL_WORKERS  questions retrieved in parallel (default 50)
#   UNH_GPU                GPU for voyage-4-nano / the cross-encoder (default 1)
set -euo pipefail
source "$(dirname "$0")/../common/env.sh"
cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="${UNH_GPU:-1}"
INPUTS="${GMEDICAL_INPUTS:-$REPO_ROOT/evaluation_data/gmedical/inputs}"
OUT="$EVAL_ROOT/unh/outputs/gmedical"
MAIN=("$PYTHON" "$EVAL_ROOT/unh/main.py" --dataset gmedical --inputs "$INPUTS" --out-dir "$OUT"
      --workers "${UNH_RETRIEVAL_WORKERS:-50}")

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
  "$PYTHON" "$EVAL_ROOT/common/answer.py" --dataset gmedical --system unh --questions "$INPUTS/questions.json" \
    --retrieval "$ctx" --out "$OUT/answers_${sys}_gmedical.json" --workers "$(scale_concurrency 16)"
  "$PYTHON" "$EVAL_ROOT/common/score_gmedical.py" --system "$sys" --questions "$INPUTS/questions.json" \
    --predictions "$OUT/answers_${sys}_gmedical.json" --out "$OUT/scores_${sys}_gmedical.json" \
    --workers "$(scale_concurrency 16)"
done
