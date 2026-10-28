#!/usr/bin/env bash
# Launch G-Medical for every system in parallel: each job runs <system>/run_gmedical.sh (own index + retrieval over
# the medical corpus as one document -> common/answer.py --dataset gmedical -> common/score_gmedical.py, LLM-Acc per
# GraphRAG-Bench question type). Logs: run_logs/<system>_gmedical.log; stop with stop_all.sh.
#
# Needs the voyage-4-nano embedding endpoint (common/env.sh), default http://localhost:8001/v1. LogicRAG also
# loads its own SentenceTransformer copy on GPU $GPU (default 0); UDH (unh) uses GPU $UNH_GPU (default 1).
# Usage: bash evaluation_scripts/launch_gmedical.sh [systems...]
#   default: hipporag2 memgraphrag logicrag hypergraphrag ms_graphrag hygrag unh
#   GRAPHRAG_CONCURRENCY (default 1200), HYGRAG_CONCURRENCY (default 500), CONCURRENCY_MULTIPLIER (common/env.sh)
set -euo pipefail
EVAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EVAL_ROOT/.." && pwd)"
if [ -z "${DEEPSEEK_API_KEY:-}" ] && [ -f "$REPO_ROOT/.env" ]; then
  DEEPSEEK_API_KEY="$(grep -E '^\s*DEEPSEEK_API_KEY\s*=' "$REPO_ROOT/.env" | head -1 | cut -d= -f2- | tr -d "\"' ")"
  export DEEPSEEK_API_KEY
fi
: "${DEEPSEEK_API_KEY:?export DEEPSEEK_API_KEY or put it in $REPO_ROOT/.env}"
export PYTHON="${PYTHON:-python}"
export GMEDICAL_INPUTS="$(cd "${GMEDICAL_INPUTS:-$REPO_ROOT/evaluation_data/gmedical/inputs}" && pwd)"
export CONCURRENCY_MULTIPLIER="${CONCURRENCY_MULTIPLIER:-1}"
if [ $# -gt 0 ]; then SYSTEMS=("$@"); else SYSTEMS=(hipporag2 memgraphrag logicrag hypergraphrag ms_graphrag hygrag unh); fi
[ -f "$GMEDICAL_INPUTS/questions.json" ] || { echo "missing $GMEDICAL_INPUTS/questions.json: run evaluation_data/build_gmedical_inputs.py"; exit 1; }
url="${EMBEDDING_BASE_URL:-http://localhost:8001/v1}"
gpu="${GPU:-0}"
curl -sf -m 30 "$url/embeddings" -H 'Content-Type: application/json' \
  -d '{"model":"voyageai/voyage-4-nano","input":["ping"]}' > /dev/null || { echo "no embedding server at $url"; exit 1; }
mkdir -p "$EVAL_ROOT/run_logs"
for sys in "${SYSTEMS[@]}"; do
  [ -f "$EVAL_ROOT/$sys/run_gmedical.sh" ] || { echo "no $sys/run_gmedical.sh"; exit 1; }
  log="$EVAL_ROOT/run_logs/${sys}_gmedical.log"
  EMBEDDING_BASE_URL="$url" CUDA_VISIBLE_DEVICES="$gpu" \
    setsid nohup bash "$EVAL_ROOT/$sys/run_gmedical.sh" > "$log" 2>&1 < /dev/null &
  echo "$!" > "$EVAL_ROOT/run_logs/${sys}_gmedical.pid"
  echo "started $sys (pid $!, GPU $gpu, embeddings $url) -> $log"
done
