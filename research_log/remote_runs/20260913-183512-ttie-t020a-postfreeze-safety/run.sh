#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260913-183512-ttie-t020a-postfreeze-safety'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260913-183512-ttie-t020a-postfreeze-safety'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260913-183512-ttie-t020a-postfreeze-safety/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
CUDA_VISIBLE_DEVICES=1 /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -u -m ttie.nonspatial_safety.evaluate --cohort /home/wenchang/asdasdsad/wjq/TTIE/runs/20260913-183032-ttie-t020a-safety-features/artifacts/cohort --selected /home/wenchang/asdasdsad/wjq/TTIE/runs/20260913-183032-ttie-t020a-safety-features/artifacts/selected --images /home/wenchang/asdasdsad/wjq/TTIE/shared/t008/val2017 --output "$AUTODL_ARTIFACTS_DIR/evaluation" --source-sha 49888ea2a8f6cb30cf292ecffb4ac715066901d2 --device cuda:0 --replay-receipt /home/wenchang/asdasdsad/wjq/TTIE/runs/20260913-183032-ttie-t020a-safety-features/artifacts/selected/T020A_replay_verification.json
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
