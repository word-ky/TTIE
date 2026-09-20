#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260921-071214-ttie-t068b-motion-knee'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260921-071214-ttie-t068b-motion-knee'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260921-071214-ttie-t068b-motion-knee/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /media/wenchang/F/wjq/TTIE/releases/20260921-ttie-t068b-motion-knee && export CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL TTIE_SOURCE_COMMIT=86b99ad7261bc9ac4b1b4fad089f616137de0d27 && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m pytest research_log/T068B research_log/T067B research_log/T063C -q && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m research_log.T068B.run --out /media/wenchang/F/wjq/TTIE/runs/T068B-dev-motion-knee && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m research_log.T068B.verify --out /media/wenchang/F/wjq/TTIE/runs/T068B-dev-motion-knee
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
