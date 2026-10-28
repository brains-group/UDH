# Settings for the Contract QA runners: every <system>/run_contracts.sh and launch_contracts.sh sources this.
#   source evaluation_scripts/common/env_contracts.sh
# The LLM setup is env.sh's (DeepSeek, thinking disabled, no temperature sent, voyage-4-nano on :8001).
_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [ -z "${DEEPSEEK_API_KEY:-}" ] && [ -f "$_repo/.env" ]; then  # optional: DEEPSEEK_API_KEY=... in <repo>/.env
  DEEPSEEK_API_KEY="$(grep -E '^\s*DEEPSEEK_API_KEY\s*=' "$_repo/.env" | head -1 | cut -d= -f2- | tr -d "\"' ")"
  export DEEPSEEK_API_KEY
fi
export LLM_API_KEY="${LLM_API_KEY:-$DEEPSEEK_API_KEY}"
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# CONTRACT_JOBS contracts are indexed / retrieved at once per system (per-system LLM concurrency is set in each
# run_contracts.sh); CONTRACT_SCORE_WORKERS parallel answer + judge calls in contracts_score.py
export CONTRACT_JOBS="${CONTRACT_JOBS:-5}"
export CONTRACT_SCORE_WORKERS="${CONTRACT_SCORE_WORKERS:-32}"
