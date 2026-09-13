#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260914-061849-ttie-t023a-validation'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-061849-ttie-t023a-validation'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-061849-ttie-t023a-validation/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /home/wenchang/asdasdsad/wjq/TTIE/shared/t023a/source && CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -m ttie.lolv2_real_core --low-root /home/wenchang/asdasdsad/wjq/TTIE/shared/t022a/low --split research_log/T022A_data/split.json --assets /home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-061503-ttie-t023a-source/artifacts/source/validation_assets.json --out "$AUTODL_ARTIFACTS_DIR/audit"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
