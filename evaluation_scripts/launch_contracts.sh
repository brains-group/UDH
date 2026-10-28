#!/usr/bin/env bash
# Launch Contract QA for every system in parallel: each job runs <system>/run_contracts.sh (one graph per contract ->
# retrieval for every question -> common/contracts_score.py: shared answer step + QA judge).
# Logs: run_logs/<system>_contracts.log (per-contract logs in run_logs/contracts/); stop with stop_all.sh.
# Results: contracts_results/long_qa/<system>/summary.json; tables: python common/collect_contracts.py
#
# Needs the voyage-4-nano embedding server (EMBEDDING_BASE_URL, default http://localhost:8001/v1).
# Usage: bash evaluation_scripts/launch_contracts.sh [systems...]
#   default: hipporag2 memgraphrag logicrag hypergraphrag ms_graphrag hygrag unh
#   CONTRACTS="a b"   some contracts only;  CONTRACT_JOBS (default 5) contracts at once per system
set -euo pipefail
EVAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SET=long_qa
if [ $# -gt 0 ]; then SYSTEMS=("$@"); else SYSTEMS=(hipporag2 memgraphrag logicrag hypergraphrag ms_graphrag hygrag unh); fi
source "$EVAL_ROOT/common/env_contracts.sh"
[ -f "$REPO_ROOT/evaluation_data/contracts/$SET/contracts.json" ] || { echo "no set $SET: run evaluation_data/build_contract_inputs.py"; exit 1; }
curl -sf -m 30 -H "Authorization: Bearer $DEEPSEEK_API_KEY" "$LLM_BASE_URL/models" > /dev/null || { echo "no LLM server at $LLM_BASE_URL"; exit 1; }
curl -sf -m 30 "$EMBEDDING_BASE_URL/embeddings" -H 'Content-Type: application/json' \
  -d '{"model":"voyageai/voyage-4-nano","input":["ping"]}' > /dev/null || { echo "no embedding server at $EMBEDDING_BASE_URL"; exit 1; }
mkdir -p "$EVAL_ROOT/run_logs"
echo "LLM $LLM_MODEL @ $LLM_BASE_URL, embeddings @ $EMBEDDING_BASE_URL, contracts: ${CONTRACTS:-all}"
for sys in "${SYSTEMS[@]}"; do
  [ -f "$EVAL_ROOT/$sys/run_contracts.sh" ] || { echo "no $sys/run_contracts.sh"; exit 1; }
  log="$EVAL_ROOT/run_logs/${sys}_contracts.log"
  setsid nohup bash "$EVAL_ROOT/$sys/run_contracts.sh" "$SET" > "$log" 2>&1 < /dev/null &
  echo "$!" > "$EVAL_ROOT/run_logs/${sys}_contracts.pid"
  echo "started $sys (pid $!) -> $log"
done
