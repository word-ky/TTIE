#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260913-175047-ttie-t019d-postfreeze-eval'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260913-175047-ttie-t019d-postfreeze-eval'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260913-175047-ttie-t019d-postfreeze-eval/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
CUDA_VISIBLE_DEVICES=1 /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -u -m ttie.fresh_deadband.evaluate --cohort /home/wenchang/asdasdsad/wjq/TTIE/runs/20260913-174415-ttie-t019d-fresh-features/artifacts/cohort --selected /home/wenchang/asdasdsad/wjq/TTIE/runs/20260913-174415-ttie-t019d-fresh-features/artifacts/selected --images /home/wenchang/asdasdsad/wjq/TTIE/shared/t008/val2017 --output "$AUTODL_ARTIFACTS_DIR/evaluation" --source-sha 2e4f8bfa81ba8edd9cd1804f724a4747f1318847 --device cuda:0 --replay-receipt /home/wenchang/asdasdsad/wjq/TTIE/runs/20260913-174415-ttie-t019d-fresh-features/artifacts/selected/T019D_replay_verification.json
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
