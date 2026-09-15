#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260915-202646-ttie-t048a-oracle'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-202646-ttie-t048a-oracle'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-202646-ttie-t048a-oracle/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH="$PWD" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python research_log/T048A_oracle/run.py --split research_log/T022A_data/split.json --low-root /home/wenchang/asdasdsad/wjq/TTIE/shared/t022a/low --normal-root /home/wenchang/asdasdsad/wjq/TTIE/shared/t028a/REFERENCE_ORACLE_ONLY/normal --accepted /home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-191837-ttie-t047a-oracle/artifacts/REFERENCE_ORACLE_ONLY --preflight /home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-202511-ttie-t048a-preflight-complete/artifacts/preflight --out "$AUTODL_RUN_DIR/artifacts/REFERENCE_ORACLE_ONLY"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
