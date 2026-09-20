#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260921-062905-ttie-t068a-step-cap'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260921-062905-ttie-t068a-step-cap'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260921-062905-ttie-t068a-step-cap/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /media/wenchang/F/wjq/TTIE/releases/20260921-ttie-t068a-step-cap && export CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL TTIE_SOURCE_COMMIT=3f0f2d5cbca0ab13efbd7ddd1bdf5570c74cbd38 && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m pytest research_log/T068A research_log/T067B research_log/T063C -q && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m research_log.T068A.run --out /media/wenchang/F/wjq/TTIE/runs/T068A-dev-step-cap && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m research_log.T068A.verify --out /media/wenchang/F/wjq/TTIE/runs/T068A-dev-step-cap
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
