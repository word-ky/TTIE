#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260915-175524-ttie-t046a-eval'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-175524-ttie-t046a-eval'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-175524-ttie-t046a-eval/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python research_log/T046A_oracle/evaluate.py --split research_log/T022A_data/split.json --normal-root /home/wenchang/asdasdsad/wjq/TTIE/shared/t028a/REFERENCE_ORACLE_ONLY/normal --accepted /home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-023649-ttie-t035a-oracle/artifacts/REFERENCE_ORACLE_ONLY --preflight /home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-173248-ttie-t046a-preflight/artifacts/preflight --out /home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-173335-ttie-t046a-oracle/artifacts/REFERENCE_ORACLE_ONLY && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python research_log/T046A_oracle/replay.py --split research_log/T022A_data/split.json --normal-root /home/wenchang/asdasdsad/wjq/TTIE/shared/t028a/REFERENCE_ORACLE_ONLY/normal --accepted /home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-023649-ttie-t035a-oracle/artifacts/REFERENCE_ORACLE_ONLY --preflight /home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-173248-ttie-t046a-preflight/artifacts/preflight --out /home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-173335-ttie-t046a-oracle/artifacts/REFERENCE_ORACLE_ONLY
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
