#!/usr/bin/env bash
set -euo pipefail

root=/media/wenchang/F/wjq/TTIE/migrations/paid4090_20260929
virtual=/root/autodl-tmp/TTIE
code=$virtual/T075A/codev2/research_log/T075A/v2
source=$virtual/T075B/runs/SID/mri
source_freeze=$virtual/T075B/freeze/SID/mr_illuminate/freeze_receipt.json
low=$virtual/T074B/targets/SID/low_receipt.json
outputs=$virtual/T075A/rows/SID/mr_illuminate_plus_D
freeze=$virtual/T075A/freeze/SID/mr_illuminate_plus_D
py=/home/wenchang/anaconda3/envs/python3.12-tk2-2.3/bin/python
export CUDA_VISIBLE_DEVICES=0

run() {
  bwrap --ro-bind / / --tmpfs /root --dir /root/autodl-tmp \
    --bind "$root" "$virtual" --proc /proc --dev-bind /dev /dev --bind /tmp /tmp \
    --chdir "$virtual/T073C/recovery/source" "$py" -B "$code/$1" "${@:2}"
}

run denoise_d.py \
  --source-dir "$source" --source-row mr_illuminate --source-freeze "$source_freeze" \
  --method-id mr_illuminate_plus_D --low-receipt "$low" --expected-count 598 \
  --workers 4 --out "$outputs"
run verify_v2_rows.py \
  --kind plus_d --method-id mr_illuminate_plus_D --low-receipt "$low" \
  --outputs "$outputs" --expected-count 598 --source-dir "$source" \
  --source-row mr_illuminate --out "$outputs.verify.json"
run make_freeze_v2.py \
  --target SID --row mr_illuminate_plus_D --outputs "$outputs" \
  --source-freeze "$source_freeze" --out "$freeze" \
  --path-prefix research_log/T075B/SID/mr_illuminate_plus_D
