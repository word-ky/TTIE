#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260921-042913-ttie-t067c-interior'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260921-042913-ttie-t067c-interior'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260921-042913-ttie-t067c-interior/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /home/wenchang/asdasdsad/wjq/TTIE/releases/20260921-ttie-t067c-interior && export CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL TTIE_SOURCE_COMMIT=21aaceb5e395749fb06d80fd019aafdbc12a9e1f && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m pytest research_log/T067B/test_core.py research_log/T067C/test_core.py -q && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m research_log.T067C.run --out /media/wenchang/F/wjq/TTIE/runs/T067C-interior-transfer && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m research_log.T067C.verify --out /media/wenchang/F/wjq/TTIE/runs/T067C-interior-transfer
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
