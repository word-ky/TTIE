#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260923-181604-ttie-t072ad-sealed-smoke'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260923-181604-ttie-t072ad-sealed-smoke'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260923-181604-ttie-t072ad-sealed-smoke/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /home/wenchang/asdasdsad/wjq/TTIE/releases/20260923-050852-ttie-t072q-sealed-smoke && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python research_log/T072O/launcher.py --runtime-root /media/wenchang/F/wjq/TTIE/releases/20260921-ttie-t071b-official-baselines --out /media/wenchang/F/wjq/TTIE/runs/T072AD-sealed-smoke-once
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
