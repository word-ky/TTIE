#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260916-032631-ttie-t051a-replay'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260916-032631-ttie-t051a-replay'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260916-032631-ttie-t051a-replay/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python /home/wenchang/asdasdsad/wjq/TTIE/shared/t051a/replay_corrected.py --split research_log/T022A_data/split.json --normal-root /home/wenchang/asdasdsad/wjq/TTIE/shared/t028a/REFERENCE_ORACLE_ONLY/normal --accepted /home/wenchang/asdasdsad/wjq/TTIE/runs/20260916-004508-ttie-t050a-oracle/artifacts/REFERENCE_ORACLE_ONLY --preflight /home/wenchang/asdasdsad/wjq/TTIE/runs/20260916-032032-ttie-t051a-preflight/artifacts/preflight --out /home/wenchang/asdasdsad/wjq/TTIE/runs/20260916-032119-ttie-t051a-oracle/artifacts/REFERENCE_ORACLE_ONLY --low-root /home/wenchang/asdasdsad/wjq/TTIE/shared/t022a/low
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
