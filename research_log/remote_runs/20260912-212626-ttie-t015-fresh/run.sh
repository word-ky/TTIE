#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260912-212626-ttie-t015-fresh'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260912-212626-ttie-t015-fresh'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260912-212626-ttie-t015-fresh/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
env AUTODL_ARTIFACTS_DIR="/media/wenchang/F/wjq/TTIE/runs/$AUTODL_RUN_ID/artifacts" CUDA_VISIBLE_DEVICES=1 bash scripts/run_t015_a6000.sh c4e58e5ad64bfce0bea72561997db8007e12b510
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
