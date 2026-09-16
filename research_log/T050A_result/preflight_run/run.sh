#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260916-004324-ttie-t050a-preflight'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260916-004324-ttie-t050a-preflight'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260916-004324-ttie-t050a-preflight/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH="$PWD" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -m pytest tests/test_t049a_tone.py tests/test_t050a_continuation.py -q && CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH="$PWD" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python research_log/T050A_oracle/preflight.py --split research_log/T022A_data/split.json --low-root /home/wenchang/asdasdsad/wjq/TTIE/shared/t022a/low --accepted /home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-212417-ttie-t049a-oracle/artifacts/REFERENCE_ORACLE_ONLY --prior-preflight /home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-212322-ttie-t049a-preflight/artifacts/preflight --binding research_log/T050A_source_binding.json --out "$AUTODL_RUN_DIR/artifacts/preflight"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
