#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260920-041332-ttie-t062a-zr'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260920-041332-ttie-t062a-zr'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260920-041332-ttie-t062a-zr/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /home/wenchang/asdasdsad/wjq/TTIE/releases/20260920-ttie-t062a-zr
export CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python
OUT=/media/wenchang/F/wjq/TTIE/runs/T062A-fixed-zr
$PY -m pytest -p no:cacheprovider --import-mode=importlib research_log/T062A/test_core.py -q && $PY -m research_log.T062A.infer --out "$OUT" && $PY -m research_log.T062A.evaluate --out "$OUT"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
