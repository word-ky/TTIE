#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260921-172057-ttie-t071a-official-real'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260921-172057-ttie-t071a-official-real'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260921-172057-ttie-t071a-official-real/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /media/wenchang/F/wjq/TTIE/releases/20260921-ttie-t071a-official-real && export CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL TTIE_SOURCE_COMMIT=cdabd083ad0853b702f409c4115f73f657d89918 && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m pytest research_log/T071A research_log/T070A -q && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m research_log.T071A.run --out /media/wenchang/F/wjq/TTIE/runs/T071A-official-real && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m research_log.T071A.evaluate --out /media/wenchang/F/wjq/TTIE/runs/T071A-official-real && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m research_log.T071A.verify --out /media/wenchang/F/wjq/TTIE/runs/T071A-official-real
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
