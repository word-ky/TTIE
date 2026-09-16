#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260917-011247-ttie-t057a-eval'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260917-011247-ttie-t057a-eval'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260917-011247-ttie-t057a-eval/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$PWD"
/home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python research_log/T057A_oracle/evaluate.py --split research_log/T022A_data/split.json --normal-root /home/wenchang/asdasdsad/wjq/TTIE/shared/t028a/REFERENCE_ORACLE_ONLY/normal --accepted /home/wenchang/asdasdsad/wjq/TTIE/runs/20260916-205307-ttie-t055a-oracle/artifacts/REFERENCE_ORACLE_ONLY --preflight /home/wenchang/asdasdsad/wjq/TTIE/runs/20260917-010902-ttie-t057a-preflight/artifacts/preflight --out /home/wenchang/asdasdsad/wjq/TTIE/runs/20260917-010959-ttie-t057a-oracle/artifacts/REFERENCE_ORACLE_ONLY && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python research_log/T057A_oracle/replay.py --split research_log/T022A_data/split.json --low-root /home/wenchang/asdasdsad/wjq/TTIE/shared/t022a/low --normal-root /home/wenchang/asdasdsad/wjq/TTIE/shared/t028a/REFERENCE_ORACLE_ONLY/normal --accepted /home/wenchang/asdasdsad/wjq/TTIE/runs/20260916-205307-ttie-t055a-oracle/artifacts/REFERENCE_ORACLE_ONLY --preflight /home/wenchang/asdasdsad/wjq/TTIE/runs/20260917-010902-ttie-t057a-preflight/artifacts/preflight --out /home/wenchang/asdasdsad/wjq/TTIE/runs/20260917-010959-ttie-t057a-oracle/artifacts/REFERENCE_ORACLE_ONLY
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
