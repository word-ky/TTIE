#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260920-210459-ttie-t066a-dynamics'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260920-210459-ttie-t066a-dynamics'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260920-210459-ttie-t066a-dynamics/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /home/wenchang/asdasdsad/wjq/TTIE/releases/20260920-ttie-t066a-dynamics && export CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL TTIE_SOURCE_COMMIT=c6d7a40a78e35a85befb0d77b3bbd103350f3387 && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m pytest research_log/T066A/test_core.py research_log/T065B/test_core.py research_log/T065A/test_core.py research_log/T063C/test_core.py research_log/T063A/test_core.py -q && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m research_log.T066A.run --out /media/wenchang/F/wjq/TTIE/runs/T066A-dynamics-safety && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m research_log.T066A.verify --out /media/wenchang/F/wjq/TTIE/runs/T066A-dynamics-safety
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
