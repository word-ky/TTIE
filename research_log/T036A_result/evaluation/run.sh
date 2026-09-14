#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260915-040317-ttie-t036a-eval'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-040317-ttie-t036a-eval'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-040317-ttie-t036a-eval/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
export PYTHONPATH="$PWD"
/home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python scripts/evaluate_t036a.py --audit /home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-035341-ttie-t036a-common/artifacts/audit --manifest research_log/T036A_cohort/manifest.json --normal-root /home/wenchang/asdasdsad/wjq/TTIE/shared/t036a/normal --deployment /home/wenchang/asdasdsad/wjq/TTIE/shared/t036a/reference_deployment.json
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
