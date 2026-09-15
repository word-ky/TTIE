#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260915-085324-ttie-t039a-stage-b'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-085324-ttie-t039a-stage-b'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-085324-ttie-t039a-stage-b/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
export CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH="$PWD" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python research_log/T039A_tangent/stage_b.py --stage-a /home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-083056-ttie-t039a-stage-a/artifacts/stage_a --manifest research_log/T014_source_manifest.json --images /home/wenchang/asdasdsad/wjq/TTIE/shared/t008/val2017 --out "$AUTODL_RUN_DIR/artifacts/stage_b"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
