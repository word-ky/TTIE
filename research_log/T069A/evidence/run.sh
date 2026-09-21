#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260921-115710-ttie-t069a-projection-pressure'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260921-115710-ttie-t069a-projection-pressure'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260921-115710-ttie-t069a-projection-pressure/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /media/wenchang/F/wjq/TTIE/releases/20260921-ttie-t069a-projection-pressure && export CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL TTIE_SOURCE_COMMIT=d19973d9058844cc1ec54639f61e77efc6cbb13d && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m pytest research_log/T069A research_log/T068D research_log/T067B -q && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m research_log.T069A.run --out /media/wenchang/F/wjq/TTIE/runs/T069A-projection-pressure && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m research_log.T069A.verify --out /media/wenchang/F/wjq/TTIE/runs/T069A-projection-pressure
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
