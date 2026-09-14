#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260915-005425-ttie-t033a-eval'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-005425-ttie-t033a-eval'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-005425-ttie-t033a-eval/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
export PYTHONPATH="$PWD"
/home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python scripts/evaluate_t033a.py --audit /home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-005309-ttie-t033a-retinex/artifacts/audit --split research_log/T022A_data/split.json --normal-root /home/wenchang/asdasdsad/wjq/TTIE/shared/t022a/normal --deployment /home/wenchang/asdasdsad/wjq/TTIE/shared/t033a/evaluation_deployment.json --ours research_log/remote_runs/20260914-113853-ttie-t026a-gamma05/artifacts/audit/metrics.csv --binding research_log/T033A/binding.json
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
