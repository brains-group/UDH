#!/usr/bin/env bash
# Launch HotpotQA and MuSiQue for every system in parallel. Each job runs its system's run.sh <dataset> (own index +
# retrieval -> common/answer.py -> common/score.py with the LLM judge). Logs: run_logs/<system>_<dataset>.log
#
# Needs one voyage-4-nano embedding endpoint (see common/env.sh for the command), by default
# http://localhost:8001/v1. Jobs run on GPU $GPU (default 0): LogicRAG loads its own SentenceTransformer copy of
# voyage-4-nano; UDH (unh) loads voyage-4-nano and the cross-encoder on GPU $UNH_GPU (default 1). Each job gets its
# own process group (setsid + nohup), so it keeps running after the terminal closes; stop them with stop_all.sh.
# Usage: bash evaluation_scripts/launch_all.sh [systems...]
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
export JUDGE="${JUDGE-1}"
if [ $# -gt 0 ]; then SYSTEMS=("$@"); else SYSTEMS=(hipporag2 memgraphrag logicrag hypergraphrag ms_graphrag hygrag unh); fi
url="${EMBEDDING_BASE_URL:-http://localhost:8001/v1}"
gpu="${GPU:-0}"
curl -sf -m 30 "$url/embeddings" -H 'Content-Type: application/json' \
  -d '{"model":"voyageai/voyage-4-nano","input":["ping"]}' > /dev/null || { echo "no embedding server at $url"; exit 1; }
mkdir -p "$EVAL_ROOT/run_logs"
for ds in hotpotqa musique; do
  for sys in "${SYSTEMS[@]}"; do
    log="$EVAL_ROOT/run_logs/${sys}_${ds}.log"
    EMBEDDING_BASE_URL="$url" CUDA_VISIBLE_DEVICES="$gpu" \
      setsid nohup bash "$EVAL_ROOT/$sys/run.sh" "$ds" > "$log" 2>&1 < /dev/null &
    echo "$!" > "$EVAL_ROOT/run_logs/${sys}_${ds}.pid"
    echo "started $sys $ds (pid $!, GPU $gpu, embeddings $url) -> $log"
  done
done
