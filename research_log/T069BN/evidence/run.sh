#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260921-135734-ttie-t069bn-gradient-numerics'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260921-135734-ttie-t069bn-gradient-numerics'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260921-135734-ttie-t069bn-gradient-numerics/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /media/wenchang/F/wjq/TTIE/releases/20260921-ttie-t069bn-gradient-numerics && export CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL TTIE_SOURCE_COMMIT=afc6fbeb60ef6debb777b2181480c323fe81313d && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m pytest research_log/T069BN research_log/T069B research_log/T067B -q && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m research_log.T069BN.run --out /media/wenchang/F/wjq/TTIE/runs/T069BN-gradient-numerics && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -B -m research_log.T069BN.verify --out /media/wenchang/F/wjq/TTIE/runs/T069BN-gradient-numerics
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
