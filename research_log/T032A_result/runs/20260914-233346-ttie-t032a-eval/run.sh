#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260914-233346-ttie-t032a-eval'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-233346-ttie-t032a-eval'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-233346-ttie-t032a-eval/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
export PYTHONPATH="$PWD"
PY=/home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python
AUDIT=/home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-232831-ttie-t032a-trust/artifacts/audit
$PY scripts/evaluate_t030a.py --audit "$AUDIT" --manifest research_log/T032A_cohort/manifest.json --normal-root /home/wenchang/asdasdsad/wjq/TTIE/shared/t032a/normal --deployment /home/wenchang/asdasdsad/wjq/TTIE/shared/t032a/reference_deployment.json && $PY scripts/summarize_t032a.py --audit "$AUDIT"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
