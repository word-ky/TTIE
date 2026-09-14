#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260915-055515-ttie-t037a-evaluate'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-055515-ttie-t037a-evaluate'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-055515-ttie-t037a-evaluate/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python research_log/T037A_diagnostic/evaluate.py --accepted /home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-035341-ttie-t036a-common/artifacts/audit --manifest research_log/T036A_cohort/manifest.json --normal-root /home/wenchang/asdasdsad/wjq/TTIE/shared/t036a/normal --out /home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-055148-ttie-t037a-reconstruct/artifacts/REFERENCE_DIAGNOSTIC_ONLY
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
