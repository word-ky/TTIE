#!/usr/bin/env bash
set -euo pipefail

root=/media/wenchang/F/wjq/TTIE/migrations/paid4090_20260929
virtual=/root/autodl-tmp/TTIE
python_bin=${SID_RESTORE_PYTHON:-/home/wenchang/anaconda3/envs/dl/bin/python}
export CUDA_VISIBLE_DEVICES=0

bwrap --ro-bind / / --tmpfs /root --dir /root/autodl-tmp \
  --bind "$root" "$virtual" --proc /proc --dev-bind /dev /dev --bind /tmp /tmp \
  --chdir "$virtual/T073C/recovery/source" \
  "$python_bin" -B \
  "$virtual/T075A/codev2/research_log/T075A/v2/regen_rows.py" restore \
  --outputs "$virtual/T075A/rows/SID/$1" \
  --source-dir "$virtual/T075B/runs/SID/$2"
