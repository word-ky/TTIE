#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260915-023649-ttie-t035a-oracle'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-023649-ttie-t035a-oracle'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-023649-ttie-t035a-oracle/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
export CUDA_VISIBLE_DEVICES=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$PWD"
PY=/home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python
ACCEPTED=/home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-113853-ttie-t026a-gamma05/artifacts/audit
PRIOR=/home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-163238-ttie-t028a-oracle/artifacts/REFERENCE_ORACLE_ONLY
FULL=/home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-014747-ttie-t034a-oracle/artifacts/REFERENCE_ORACLE_ONLY
PRE=/home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-023543-ttie-t035a-preflight/artifacts/preflight
LOW=/home/wenchang/asdasdsad/wjq/TTIE/shared/t022a/low
NORMAL=/home/wenchang/asdasdsad/wjq/TTIE/shared/t028a/REFERENCE_ORACLE_ONLY/normal
OUT="$AUTODL_ARTIFACTS_DIR/REFERENCE_ORACLE_ONLY"
$PY research_log/T035A_oracle/run.py --split research_log/T022A_data/split.json --low-root "$LOW" --normal-root "$NORMAL" --accepted "$ACCEPTED" --preflight "$PRE" --out "$OUT" &&
$PY research_log/T035A_oracle/evaluate.py --split research_log/T022A_data/split.json --normal-root "$NORMAL" --prior-oracle "$PRIOR" --full-wb "$FULL" --preflight "$PRE" --out "$OUT"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
