#!/usr/bin/env bash
set -uo pipefail
cd '/home/wenchang/asdasdsad/wjq/TTIE/current'
export AUTODL_RUN_ID='20260917-105829-ttie-t058ac-copy'
export AUTODL_RUN_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260917-105829-ttie-t058ac-copy'
export AUTODL_ARTIFACTS_DIR='/home/wenchang/asdasdsad/wjq/TTIE/runs/20260917-105829-ttie-t058ac-copy/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
export CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH="$PWD" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python -m pytest research_log/T058AC/test_cuda_comparison.py research_log/T058AC/test_chain.py research_log/T058AA/test_decomposition.py tests/test_t058z_cast.py tests/test_t058a_detail.py -q && /home/wenchang/asdasdsad/wjq/TTIE/.venv/bin/python research_log/T058AC/run.py --bank-root /home/wenchang/asdasdsad/wjq/TTIE/runs/20260912-181047-ttie-t014-stage-a-repaired/artifacts/audit --receipt research_log/T039A_bound_T014_receipt.json --manifest research_log/T014_source_manifest.json --checkpoint /home/wenchang/asdasdsad/wjq/TTIE/research_log/T014_energy.pt --prototypes /home/wenchang/asdasdsad/wjq/TTIE/research_log/remote_runs/20260912-071126-ttie-t006-a6000/artifacts/audit/prototypes.pt --binding research_log/T058AC_source_binding.json --canonical-selection research_log/T039A_result/stage_a/selection.json --stopped-run /home/wenchang/asdasdsad/wjq/TTIE/runs/20260917-023631-ttie-t058a-stage-a --out "$AUTODL_ARTIFACTS_DIR/T058AC"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
