#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260914-163238-ttie-t028a-oracle'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-163238-ttie-t028a-oracle'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-163238-ttie-t028a-oracle/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
set -e
export CUDA_VISIBLE_DEVICES=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$PWD"
PY=/home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python
ROOT=/home/wenchang/asdasdsad/wjq/TTIE/shared/t028a/REFERENCE_ORACLE_ONLY
ACCEPTED=/home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-113853-ttie-t026a-gamma05/artifacts/audit
OUT="$AUTODL_ARTIFACTS_DIR/REFERENCE_ORACLE_ONLY"
$PY research_log/T028A_oracle/run.py --split research_log/T022A_data/split.json --low-root /home/wenchang/asdasdsad/wjq/TTIE/shared/t022a/low --normal-root "$ROOT/normal" --accepted "$ACCEPTED" --preflight "$ROOT/preflight" --out "$OUT"
$PY research_log/T028A_oracle/evaluate.py --split research_log/T022A_data/split.json --low-root /home/wenchang/asdasdsad/wjq/TTIE/shared/t022a/low --normal-root "$ROOT/normal" --accepted "$ACCEPTED" --preflight "$ROOT/preflight" --out "$OUT"
$PY research_log/T028A_oracle/aggregate.py "$OUT"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
