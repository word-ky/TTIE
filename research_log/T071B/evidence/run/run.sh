#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260921-184705-ttie-t071b-official-baselines'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260921-184705-ttie-t071b-official-baselines'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260921-184705-ttie-t071b-official-baselines/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /media/wenchang/F/wjq/TTIE/releases/20260921-ttie-t071b-official-baselines && export CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL PYTHONPATH=$PWD TTIE_SOURCE_COMMIT=579c3691a80f5b7cadfd706a2fe6750876c53aa0 && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m research_log.T071B.run --method retinexformer --out /media/wenchang/F/wjq/TTIE/runs/T071B-official-baselines/retinexformer && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m research_log.T071B.run --method snr_aware --out /media/wenchang/F/wjq/TTIE/runs/T071B-official-baselines/snr_aware && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m research_log.T071B.evaluate --out /media/wenchang/F/wjq/TTIE/runs/T071B-official-baselines && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m research_log.T071B.verify --out /media/wenchang/F/wjq/TTIE/runs/T071B-official-baselines
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
