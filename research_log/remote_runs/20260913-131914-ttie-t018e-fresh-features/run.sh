#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260913-131914-ttie-t018e-fresh-features'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260913-131914-ttie-t018e-fresh-features'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260913-131914-ttie-t018e-fresh-features/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
bash scripts/run_t018e_a6000.sh f0403a344233c0f082d7cf9a6c7f54ba63011efb
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
