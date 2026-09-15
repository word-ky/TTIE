#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID=20260915-152656-ttie-t044a-stage-b
export AUTODL_RUN_DIR=/home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-152656-ttie-t044a-stage-b
export AUTODL_ARTIFACTS_DIR=/home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-152656-ttie-t044a-stage-b/artifacts
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python research_log/T044A_audit/stage_b.py --stage-a /home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-152458-ttie-t044a-stage-a/artifacts/stage_a --metrics /home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-035341-ttie-t036a-common/artifacts/audit/metrics.csv --out "$AUTODL_RUN_DIR/artifacts/stage_b" && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python research_log/T044A_audit/replay.py --stage-a /home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-152458-ttie-t044a-stage-a/artifacts/stage_a --stage-b "$AUTODL_RUN_DIR/artifacts/stage_b" --accepted /home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-035341-ttie-t036a-common/artifacts/audit --metrics /home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-035341-ttie-t036a-common/artifacts/audit/metrics.csv
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
