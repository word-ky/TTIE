#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260920-092400-ttie-t062cr2-fresh'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260920-092400-ttie-t062cr2-fresh'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260920-092400-ttie-t062cr2-fresh/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /home/wenchang/asdasdsad/wjq/TTIE/releases/20260920-ttie-t062cr2-fresh
export CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH="$PWD:$PWD/tests" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TTIE_SOURCE_COMMIT=47bcab6bc9cbc23712ce503f758ccad6a57f240d
/home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -m research_log.T062CR2.infer --out /media/wenchang/F/wjq/TTIE/runs/T062CR2-fresh-step27 && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -m research_log.T062CR2.evaluate --out /media/wenchang/F/wjq/TTIE/runs/T062CR2-fresh-step27 && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -m research_log.T062CR2.verify --out /media/wenchang/F/wjq/TTIE/runs/T062CR2-fresh-step27
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
