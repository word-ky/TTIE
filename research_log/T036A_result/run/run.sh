#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260915-035341-ttie-t036a-common'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-035341-ttie-t036a-common'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-035341-ttie-t036a-common/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$PWD"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python
$PY scripts/run_t036a.py --low-root /home/wenchang/asdasdsad/wjq/TTIE/shared/t036a/low --manifest research_log/T036A_cohort/manifest.json --assets research_log/T036A_assets.json --preflight /home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-035138-ttie-t036a-preflight/artifacts/preflight.json --out "$AUTODL_ARTIFACTS_DIR/audit" &&
$PY scripts/replay_t036a.py --audit "$AUTODL_ARTIFACTS_DIR/audit"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
