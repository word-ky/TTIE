#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260919-044847-ttie-t059x-eval-only'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260919-044847-ttie-t059x-eval-only'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260919-044847-ttie-t059x-eval-only/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /home/wenchang/asdasdsad/wjq/TTIE/releases/20260919-ttie-t059x-real
export CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -m research_log.T059X.evaluate_raw --out /home/wenchang/asdasdsad/wjq/TTIE/runs/20260919-044358-ttie-t059x-real/artifacts/T059X && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -m research_log.T059X.verify_raw --out /home/wenchang/asdasdsad/wjq/TTIE/runs/20260919-044358-ttie-t059x-real/artifacts/T059X
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
