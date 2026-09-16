#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260917-062907-ttie-t058y-shadow'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260917-062907-ttie-t058y-shadow'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260917-062907-ttie-t058y-shadow/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
export CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH="$PWD" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -m pytest tests/test_t058y_shadow.py tests/test_t058a_detail.py -q && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python research_log/T058Y/run.py --bank-root /home/wenchang/asdasdsad/wjq/TTIE/runs/20260912-181047-ttie-t014-stage-a-repaired/artifacts/audit --receipt research_log/T039A_bound_T014_receipt.json --manifest research_log/T014_source_manifest.json --checkpoint /home/wenchang/asdasdsad/wjq/TTIE/research_log/T014_energy.pt --prototypes /home/wenchang/asdasdsad/wjq/TTIE/research_log/remote_runs/20260912-071126-ttie-t006-a6000/artifacts/audit/prototypes.pt --binding research_log/T058Y_source_binding.json --canonical-selection research_log/T039A_result/stage_a/selection.json --stopped-run /home/wenchang/asdasdsad/wjq/TTIE/runs/20260917-023631-ttie-t058a-stage-a --out "$AUTODL_ARTIFACTS_DIR/T058Y"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
