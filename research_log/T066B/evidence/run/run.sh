#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260920-224647-ttie-t066b-support-r1'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260920-224647-ttie-t066b-support-r1'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260920-224647-ttie-t066b-support-r1/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /home/wenchang/asdasdsad/wjq/TTIE/releases/20260920-ttie-t066b-support && export CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL TTIE_SOURCE_COMMIT=e4c011d2bbd183f6227895e6ac3b940aacb6404b && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m pytest research_log/T066A/test_core.py research_log/T066B/test_core.py -q && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m research_log.T066B.run --out /media/wenchang/F/wjq/TTIE/runs/T066B-support-diagnosis-r1 && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m research_log.T066B.verify --out /media/wenchang/F/wjq/TTIE/runs/T066B-support-diagnosis-r1
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
