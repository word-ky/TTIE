#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260921-164629-ttie-t070a-final-ours'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260921-164629-ttie-t070a-final-ours'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260921-164629-ttie-t070a-final-ours/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /media/wenchang/F/wjq/TTIE/releases/20260921-ttie-t070a-final-ours && export CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL TTIE_SOURCE_COMMIT=aa4d920dff4b5b76751c24266e95ac9696d55d90 && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m pytest research_log/T070A research_log/T067B research_log/T062CR2 -q && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m research_log.T070A.manifest --out /media/wenchang/F/wjq/TTIE/shared/t070a/FINAL_OURS_MANIFEST.json && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m research_log.T070A.audit --manifest /media/wenchang/F/wjq/TTIE/shared/t070a/FINAL_OURS_MANIFEST.json --out /media/wenchang/F/wjq/TTIE/runs/T070A-final-ours && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m research_log.T070A.verify --manifest /media/wenchang/F/wjq/TTIE/shared/t070a/FINAL_OURS_MANIFEST.json --out /media/wenchang/F/wjq/TTIE/runs/T070A-final-ours
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
