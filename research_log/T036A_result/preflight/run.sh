#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260915-035138-ttie-t036a-preflight'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-035138-ttie-t036a-preflight'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-035138-ttie-t036a-preflight/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$PWD"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python
$PY -m pytest -q tests/test_gamma_range.py tests/test_common_gain_ttt.py &&
$PY scripts/preflight_t036a.py --accepted /home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-113853-ttie-t026a-gamma05/artifacts/audit --low-root /home/wenchang/asdasdsad/wjq/TTIE/shared/t022a/low --split research_log/T022A_data/split.json --assets research_log/T036A_assets.json --binding research_log/T036A_source_binding.json --out "$AUTODL_ARTIFACTS_DIR/preflight.json"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
