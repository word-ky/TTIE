#!/usr/bin/env bash
set -euo pipefail

root=/media/wenchang/F/wjq/TTIE/migrations/paid4090_20260929
virtual=/root/autodl-tmp/TTIE
code=$virtual/T075A/gatetree/research_log/T075A/v2/targets_sid.py
base=$virtual/T075B/freeze/SID
v2=$virtual/T075A/freeze/SID
rows=$virtual/T075A/gatetree/research_log/T075B/SID
target=$virtual/T074B/targets/SID
gate=$rows/reference_gate_receipt.json
metrics=$virtual/T075A/ceiling/SID_v2_metrics
py=/home/wenchang/anaconda3/envs/python3.12-tk2-2.3/bin/python
export CUDA_VISIBLE_DEVICES=0

mkdir -p "$root/T075A/gatetree/research_log/T075B/SID"
for dir in "$root/T075B/freeze/SID"/* "$root/T075A/freeze/SID"/*; do
  cp -a "$dir" "$root/T075A/gatetree/research_log/T075B/SID/"
done

run() {
  bwrap --ro-bind / / --tmpfs /root --dir /root/autodl-tmp \
    --bind "$root" "$virtual" --proc /proc --dev-bind /dev /dev --bind /tmp /tmp \
    --chdir "$virtual/T075A/gatetree/research_log/T075A/v2" \
    "$py" -B "$code" gm "$@"
}

run gate --target SID --low-receipt "$target/low_receipt.json" \
  --opaque-manifest "$target/reference_opaque_manifest.json" \
  --rows-dir "$rows" --out "$gate"
run metrics --target SID --low-receipt "$target/low_receipt.json" \
  --opaque-manifest "$target/reference_opaque_manifest.json" \
  --rows-dir "$rows" --gate-receipt "$gate" --out "$metrics" --workers 4
