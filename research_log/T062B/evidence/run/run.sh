#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260920-054721-ttie-t062b-audit'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260920-054721-ttie-t062b-audit'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260920-054721-ttie-t062b-audit/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /home/wenchang/asdasdsad/wjq/TTIE/releases/20260920-ttie-t062b-audit
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python
OUT=/media/wenchang/F/wjq/TTIE/runs/T062B-frozen-audit
$PY -m pytest -p no:cacheprovider --import-mode=importlib research_log/T062B/test_selection.py -q && $PY -m research_log.T062B.analyze --out "$OUT" && $PY -m research_log.T062B.verify --out "$OUT"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
