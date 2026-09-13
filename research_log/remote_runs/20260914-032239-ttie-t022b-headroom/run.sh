#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260914-032239-ttie-t022b-headroom'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-032239-ttie-t022b-headroom'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-032239-ttie-t022b-headroom/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /home/wenchang/asdasdsad/wjq/TTIE/shared/t022b/source && CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -m ttie.trajectory_headroom --original /home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-024025-ttie-t022a-core/artifacts/audit --low-root /home/wenchang/asdasdsad/wjq/TTIE/shared/t022a/low --normal-root /home/wenchang/asdasdsad/wjq/TTIE/shared/t022a/normal --split research_log/T022A_data/split.json --freeze research_log/T022A_pre_reference_freeze.json --out "$AUTODL_ARTIFACTS_DIR/audit" && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python scripts/verify_t022b.py "$AUTODL_ARTIFACTS_DIR/audit" /home/wenchang/asdasdsad/wjq/TTIE/runs/20260914-024025-ttie-t022a-core/artifacts/audit
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
