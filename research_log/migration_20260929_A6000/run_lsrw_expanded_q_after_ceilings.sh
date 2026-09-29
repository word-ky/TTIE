#!/usr/bin/env bash
set -euo pipefail

root=/media/wenchang/F/wjq/TTIE/migrations/paid4090_20260929
virtual=/root/autodl-tmp/TTIE
queue_log=/media/wenchang/F/wjq/TTIE/runs/20260929-183549-ttie-remaining-ceilings-gpu0/train.log

while ! grep -q '^\[autodl\] exit_code=' "$queue_log"; do
  sleep 60
done
grep -q '^\[autodl\] exit_code=0$' "$queue_log"

src=$virtual/T073C/recovery/source
code=$virtual/T075A/codev2/research_log/T075A/v2/ceiling.py
target=$virtual/T074B/targets/LSRW
rows=$virtual/T075A/gatetree/research_log/T075B/LSRW
ceiling=$virtual/T075A/ceiling/LSRW/expanded_q
work=$ceiling/tuning
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

tuning_args=(--target LSRW --low-receipt "$target/low_receipt.json" \
  --low-dir "$target/low" --opaque-manifest "$target/reference_opaque_manifest.json" \
  --gate-receipt "$rows/reference_gate_receipt.json" --rows-dir "$rows" \
  --manifest "$manifest" --frozen-manifest "$frozen" --work "$work")
base_args=(--target LSRW --low-receipt "$target/low_receipt.json" \
  --opaque-manifest "$target/reference_opaque_manifest.json" \
  --gate-receipt "$rows/reference_gate_receipt.json" --rows-dir "$rows")

printf '[expanded-q] prior GPU0 ceiling queue succeeded at %s\n' "$(date -Is)"
run "$gpu_py" tuning "${tuning_args[@]}" --grid "$virtual/expanded_q_grid.json"
run "$gpu_py" tuning "${tuning_args[@]}" --round2
run "$gpu_py" tuning "${tuning_args[@]}" --materialize
run "$cpu_py" kappa "${base_args[@]}" \
  --tuned-row "$work/ours_ttt_target_tuned" --out "$ceiling/row" --workers 4
run "$cpu_py" metrics "${base_args[@]}" \
  --tuned-row "$work/ours_ttt_target_tuned" --ceiling "$ceiling/row" \
  --out "$ceiling/metrics" --workers 4 \
  --reuse-scores "$virtual/lsrw_metrics_v2_result.json"
printf '[expanded-q] separate post-hoc LSRW result complete at %s\n' "$(date -Is)"
