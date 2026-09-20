#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260921-013043-ttie-t067a-first-safe'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260921-013043-ttie-t067a-first-safe'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260921-013043-ttie-t067a-first-safe/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /home/wenchang/asdasdsad/wjq/TTIE/releases/20260921-ttie-t067a-first-safe && export CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL TTIE_SOURCE_COMMIT=e20592490094fc43f9a9692a8b7be327c20ecda1 && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m pytest research_log/T066C/test_core.py research_log/T067A/test_core.py -q && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m research_log.T067A.run --out /media/wenchang/F/wjq/TTIE/runs/T067A-first-safe && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m research_log.T067A.verify --out /media/wenchang/F/wjq/TTIE/runs/T067A-first-safe
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
