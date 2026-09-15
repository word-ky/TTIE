#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260915-161312-ttie-t045a-eval'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-161312-ttie-t045a-eval'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-161312-ttie-t045a-eval/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python scripts/evaluate_t045a.py --audit /home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-161153-ttie-t045a-snr/artifacts/audit --split research_log/T022A_data/split.json --normal-root /home/wenchang/asdasdsad/wjq/TTIE/shared/t022a/normal --deployment /home/wenchang/asdasdsad/wjq/TTIE/shared/t045a/evaluation_deployment.json --ours /home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-113853-ttie-t026a-gamma05/artifacts/audit/metrics.csv --binding research_log/T045A_binding.json && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python scripts/replay_t045a.py --audit /home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-161153-ttie-t045a-snr/artifacts/audit --split research_log/T022A_data/split.json --normal-root /home/wenchang/asdasdsad/wjq/TTIE/shared/t022a/normal --deployment /home/wenchang/asdasdsad/wjq/TTIE/shared/t045a/evaluation_deployment.json --ours /home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-113853-ttie-t026a-gamma05/artifacts/audit/metrics.csv
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
