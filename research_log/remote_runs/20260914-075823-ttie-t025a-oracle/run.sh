#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260914-075823-ttie-t025a-oracle'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-075823-ttie-t025a-oracle'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-075823-ttie-t025a-oracle/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /home/wenchang/asdasdsad/wjq/TTIE/releases/20260914-075755-ttie-t025a-oracle && PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python research_log/T025A_oracle/run_oracle.py --split research_log/T022A_data/split.json --low-root /home/wenchang/asdasdsad/wjq/TTIE/shared/t022a/low --normal-root /home/wenchang/asdasdsad/wjq/TTIE/shared/t022a/normal --accepted /home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-042925-ttie-t022c-ev2/artifacts/audit --out "$AUTODL_ARTIFACTS_DIR/REFERENCE_ORACLE_ONLY" && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python research_log/T025A_oracle/aggregate.py "$AUTODL_ARTIFACTS_DIR/REFERENCE_ORACLE_ONLY"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
