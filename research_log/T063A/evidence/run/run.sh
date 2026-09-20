#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260920-101902-ttie-t063a-prefix'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260920-101902-ttie-t063a-prefix'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260920-101902-ttie-t063a-prefix/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /home/wenchang/asdasdsad/wjq/TTIE/releases/20260920-ttie-t063a-prefix
export CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH="$PWD:$PWD/tests" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TTIE_SOURCE_COMMIT=c4c58706c23c9c8972e2720eae87d294d5163986
/home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -m research_log.T063A.reconstruct --out /media/wenchang/F/wjq/TTIE/runs/T063A-prefix-oracle && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -m research_log.T063A.analyze --out /media/wenchang/F/wjq/TTIE/runs/T063A-prefix-oracle && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -m research_log.T063A.verify --out /media/wenchang/F/wjq/TTIE/runs/T063A-prefix-oracle
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
