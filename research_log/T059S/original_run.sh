#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260918-231247-ttie-t059s-one-step'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260918-231247-ttie-t059s-one-step'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260918-231247-ttie-t059s-one-step/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /home/wenchang/asdasdsad/wjq/TTIE/releases/20260918-ttie-t059s-one-step
export CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python research_log/T059S/act.py --out "$AUTODL_ARTIFACTS_DIR/T059S" && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python research_log/T059S/evaluate.py --out "$AUTODL_ARTIFACTS_DIR/T059S" && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python research_log/T059S/verify.py --out "$AUTODL_ARTIFACTS_DIR/T059S"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
