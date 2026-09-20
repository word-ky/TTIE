#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260921-045307-ttie-t067d-boundary'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260921-045307-ttie-t067d-boundary'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260921-045307-ttie-t067d-boundary/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /home/wenchang/asdasdsad/wjq/TTIE/releases/20260921-ttie-t067d-boundary && export CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL TTIE_SOURCE_COMMIT=c5195296580a6be537be9bc37aa8da9bb090707c && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m pytest research_log/T067D/test_core.py -q && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m research_log.T067D.run --out /media/wenchang/F/wjq/TTIE/runs/T067D-interval-boundary && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m research_log.T067D.verify --out /media/wenchang/F/wjq/TTIE/runs/T067D-interval-boundary
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
