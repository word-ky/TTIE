#!/usr/bin/env bash
set -euo pipefail

root=/media/wenchang/F/wjq/TTIE/migrations/paid4090_20260929
virtual=/root/autodl-tmp/TTIE
src=$virtual/T073C/recovery/source
code=$virtual/T075A/codev2/research_log/T075A/v2/ceiling.py
target=$virtual/T074B/targets/SMID
rows=$virtual/T075A/gatetree/research_log/T075B/SMID
ceiling=$virtual/T075A/ceiling/SMID
work=$ceiling/tuning_a6000
manifest=$virtual/diagnostic_manifest_A6000.json
frozen=$virtual/T073C/recovery/artifacts/T073C_execution_manifest.json
gpu_py=/home/wenchang/anaconda3/envs/dl/bin/python
cpu_py=/home/wenchang/anaconda3/envs/python3.12-tk2-2.3/bin/python
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$root/pylib
unset CUBLAS_WORKSPACE_CONFIG OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS MKL_THREADING_LAYER

run() {
  local py=$1
  shift
  bwrap --ro-bind / / --tmpfs /root --dir /root/autodl-tmp \
    --bind "$root" "$virtual" --proc /proc --dev-bind /dev /dev --bind /tmp /tmp \
    --chdir "$src" "$py" -B "$code" "$@"
}

tuning_args=(--target SMID --low-receipt "$target/low_receipt.json" \
  --low-dir "$target/low" --opaque-manifest "$target/reference_opaque_manifest.json" \
  --gate-receipt "$rows/reference_gate_receipt.json" --rows-dir "$rows" \
  --manifest "$manifest" --frozen-manifest "$frozen" --work "$work")
base_args=(--target SMID --low-receipt "$target/low_receipt.json" \
  --opaque-manifest "$target/reference_opaque_manifest.json" \
  --gate-receipt "$rows/reference_gate_receipt.json" --rows-dir "$rows")

run "$gpu_py" tuning "${tuning_args[@]}" --round2
run "$gpu_py" tuning "${tuning_args[@]}" --materialize
run "$cpu_py" kappa "${base_args[@]}" \
  --tuned-row "$work/ours_ttt_target_tuned" --out "$ceiling/row_a6000" --workers 4
run "$cpu_py" metrics "${base_args[@]}" \
  --tuned-row "$work/ours_ttt_target_tuned" --ceiling "$ceiling/row_a6000" \
  --out "$ceiling/metrics_a6000" --workers 4 \
  --reuse-scores "$virtual/T075A/ceiling/SMID_v2_metrics/metrics_v2_result.json"
