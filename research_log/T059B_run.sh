#!/usr/bin/env bash
set -uo pipefail
cd /home/wenchang/asdasdsad/wjq/TTIE/current
export AUTODL_RUN_ID=20260917-185145-ttie-t059b-dualfit
export AUTODL_RUN_DIR=/home/wenchang/asdasdsad/wjq/TTIE/runs/20260917-185145-ttie-t059b-dualfit
export AUTODL_ARTIFACTS_DIR=/home/wenchang/asdasdsad/wjq/TTIE/runs/20260917-185145-ttie-t059b-dualfit/artifacts
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
export CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -m pytest -p no:cacheprovider --import-mode=importlib research_log/T059B/test_fit.py research_log/T059A/test_cache.py -q && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python research_log/T059B/run.py --out "$AUTODL_ARTIFACTS_DIR/T059B"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
