#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260914-194331-ttie-t031a-reference'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-194331-ttie-t031a-reference'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-194331-ttie-t031a-reference/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
export CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$PWD" CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python
$PY research_log/T031A_support/reference.py --support /home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-194203-ttie-t031a-support/artifacts/support --manifest research_log/T030A_cohort/manifest.json --assets /home/wenchang/asdasdsad/wjq/TTIE/shared/t030a/assets.json --low-root /home/wenchang/asdasdsad/wjq/TTIE/shared/t030a/low --normal-root /home/wenchang/asdasdsad/wjq/TTIE/shared/t030a/normal --out "$AUTODL_ARTIFACTS_DIR/REFERENCE_GRADIENT_DIAGNOSTIC_ONLY" && $PY research_log/T031A_support/analyze.py --support /home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-194203-ttie-t031a-support/artifacts/support --reference "$AUTODL_ARTIFACTS_DIR/REFERENCE_GRADIENT_DIAGNOSTIC_ONLY"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
