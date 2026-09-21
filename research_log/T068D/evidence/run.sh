#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260921-104317-ttie-t068d-component-regret'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260921-104317-ttie-t068d-component-regret'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260921-104317-ttie-t068d-component-regret/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /media/wenchang/F/wjq/TTIE/releases/20260921-ttie-t068d-component-regret && export CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL TTIE_SOURCE_COMMIT=2c33a93377d69f83727e847faed20add5ff38e24 && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m pytest research_log/T068D research_log/T068C research_log/T067B -q && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m research_log.T068D.run --out /media/wenchang/F/wjq/TTIE/runs/T068D-component-regret && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m research_log.T068D.verify --out /media/wenchang/F/wjq/TTIE/runs/T068D-component-regret
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
