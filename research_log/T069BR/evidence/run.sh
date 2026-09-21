#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260921-145411-ttie-t069br-float64-cancellation'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260921-145411-ttie-t069br-float64-cancellation'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260921-145411-ttie-t069br-float64-cancellation/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /media/wenchang/F/wjq/TTIE/releases/20260921-ttie-t069br-float64-cancellation && export CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL TTIE_SOURCE_COMMIT=a6cd7e4074521338dc3a5a9507cca89a24cf5804 && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m pytest research_log/T069BR research_log/T069BN research_log/T067B -q && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m research_log.T069BR.run --out /media/wenchang/F/wjq/TTIE/runs/T069BR-float64-cancellation && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m research_log.T069BR.verify --out /media/wenchang/F/wjq/TTIE/runs/T069BR-float64-cancellation
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
