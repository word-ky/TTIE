#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260914-183308-ttie-t030a-guard'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-183308-ttie-t030a-guard'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-183308-ttie-t030a-guard/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
export CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH="$PWD"
PY=/home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python
$PY -m pytest tests/test_t030a_guard.py -q && $PY scripts/run_t030a.py --low-root /home/wenchang/asdasdsad/wjq/TTIE/shared/t030a/low --manifest research_log/T030A_cohort/manifest.json --assets /home/wenchang/asdasdsad/wjq/TTIE/shared/t030a/assets.json --spec research_log/T030A_cohort/selector_spec.json --out "$AUTODL_ARTIFACTS_DIR/audit" && $PY scripts/replay_t030a.py --audit "$AUTODL_ARTIFACTS_DIR/audit"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
