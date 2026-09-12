#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260913-000502-ttie-t016a-screen'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260913-000502-ttie-t016a-screen'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260913-000502-ttie-t016a-screen/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /home/wenchang/asdasdsad/wjq/TTIE/releases/20260912-235801-ttie-t016a-screen && CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -m ttie.soft_basis.screen --source-sha 05e7dcf268c7b0479e6edabdc6ec76528d5730bf --audit /media/wenchang/F/wjq/TTIE/runs/20260912-213014-ttie-t015-fresh-ready/artifacts/audit --images /home/wenchang/asdasdsad/wjq/TTIE/shared/t008/val2017 --manifest research_log/T015_manifest.json --output "$AUTODL_ARTIFACTS_DIR/audit" --device cuda:0
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
