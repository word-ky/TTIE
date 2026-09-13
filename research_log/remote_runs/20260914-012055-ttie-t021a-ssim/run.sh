#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260914-012055-ttie-t021a-ssim'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-012055-ttie-t021a-ssim'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-012055-ttie-t021a-ssim/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /home/wenchang/asdasdsad/wjq/TTIE/shared/T021A-8cfc2e92 && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -m ttie.ssim_audit --receipt research_log/T021A_outputs_verified.json --out "$AUTODL_ARTIFACTS_DIR/audit" && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python research_log/T021A_independent_replay.py "$AUTODL_ARTIFACTS_DIR/audit"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
