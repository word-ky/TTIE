#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260912-191947-ttie-t014-stage-b'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260912-191947-ttie-t014-stage-b'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260912-191947-ttie-t014-stage-b/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
env AUTODL_ARTIFACTS_DIR="/media/wenchang/F/wjq/TTIE/runs/$AUTODL_RUN_ID/artifacts" CUDA_VISIBLE_DEVICES=1 bash scripts/run_t014_a6000.sh B f861b2c6ffde6d017cb174ef8e00cb75701bf5e1
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
