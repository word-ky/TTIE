#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260914-232831-ttie-t032a-trust'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-232831-ttie-t032a-trust'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-232831-ttie-t032a-trust/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
export CUDA_VISIBLE_DEVICES=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$PWD"
PY=/home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python
RAD=/home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-232650-ttie-t032a-radius/artifacts/radius
$PY scripts/run_t032a.py --low-root /home/wenchang/asdasdsad/wjq/TTIE/shared/t032a/low --manifest research_log/T032A_cohort/manifest.json --assets /home/wenchang/asdasdsad/wjq/TTIE/shared/t032a/assets.json --radius "$RAD" --spec "$RAD/spec.json" --out "$AUTODL_ARTIFACTS_DIR/audit" && $PY scripts/replay_t032a.py --audit "$AUTODL_ARTIFACTS_DIR/audit" --radius "$RAD" --spec "$RAD/spec.json"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
