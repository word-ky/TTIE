#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260913-132546-ttie-t018e-postfreeze-eval'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260913-132546-ttie-t018e-postfreeze-eval'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260913-132546-ttie-t018e-postfreeze-eval/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
CUDA_VISIBLE_DEVICES=1 /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -u -m ttie.fresh_direction.evaluate --cohort /home/wenchang/asdasdsad/wjq/TTIE/runs/20260913-131914-ttie-t018e-fresh-features/artifacts/cohort --selected /home/wenchang/asdasdsad/wjq/TTIE/runs/20260913-131914-ttie-t018e-fresh-features/artifacts/selected --images /home/wenchang/asdasdsad/wjq/TTIE/shared/t008/val2017 --output /home/wenchang/asdasdsad/wjq/TTIE/runs/20260913-131914-ttie-t018e-fresh-features/artifacts/evaluation --source-sha f0403a344233c0f082d7cf9a6c7f54ba63011efb --device cuda:0
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
