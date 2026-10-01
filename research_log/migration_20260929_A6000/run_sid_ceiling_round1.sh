#!/usr/bin/env bash
set -euo pipefail

root=/media/wenchang/F/wjq/TTIE/migrations/paid4090_20260929
virtual=/root/autodl-tmp/TTIE
code=$virtual/T075A/codev2/research_log/T075A/v2/targets_sid.py
target=$virtual/T074B/targets/SID
rows=$virtual/T075A/gatetree/research_log/T075B/SID
work=$virtual/T075A/ceiling/SID/tuning_a6000
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$root/pylib
unset CUBLAS_WORKSPACE_CONFIG OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS MKL_THREADING_LAYER

bwrap --ro-bind / / --tmpfs /root --dir /root/autodl-tmp \
  --bind "$root" "$virtual" --proc /proc --dev-bind /dev /dev --bind /tmp /tmp \
  --chdir "$virtual/T073C/recovery/source" \
  /home/wenchang/anaconda3/envs/dl/bin/python -B "$code" ceiling tuning \
  --target SID --low-receipt "$target/low_receipt.json" --low-dir "$target/low" \
  --opaque-manifest "$target/reference_opaque_manifest.json" \
  --gate-receipt "$rows/reference_gate_receipt.json" --rows-dir "$rows" \
  --manifest "$virtual/diagnostic_manifest_A6000.json" \
  --frozen-manifest "$virtual/T073C/recovery/artifacts/T073C_execution_manifest.json" \
  --work "$work" --grid "$virtual/T075A/codev2/research_log/T074C/tuning_grid_round1.json"
