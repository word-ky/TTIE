#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260914-172718-ttie-t029a-preflight'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-172718-ttie-t029a-preflight'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-172718-ttie-t029a-preflight/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
export CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH="$PWD"
PY=/home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python
$PY -m pytest tests/test_t029a_alignment.py -q && $PY research_log/T029A_alignment/prepare.py --accepted /home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-113853-ttie-t026a-gamma05/artifacts/audit --split research_log/T022A_data/split.json --low-root /home/wenchang/asdasdsad/wjq/TTIE/shared/t022a/low --out "$AUTODL_ARTIFACTS_DIR/preflight"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
