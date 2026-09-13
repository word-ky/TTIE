#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260914-061503-ttie-t023a-source'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-061503-ttie-t023a-source'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-061503-ttie-t023a-source/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /home/wenchang/asdasdsad/wjq/TTIE/shared/t023a/source && CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -m ttie.real_source_sobolev --source-root /home/wenchang/asdasdsad/wjq/TTIE/shared/t023a/source_pairs --manifest research_log/T023A_source_manifest.json --assets research_log/T022A_assets.json --out "$AUTODL_ARTIFACTS_DIR/source"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
