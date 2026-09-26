#!/usr/bin/env bash
cd /root/autodl-tmp/TTIE/T073B/shared || exit 1
/root/miniconda3/bin/python -B run_static_native_batch.py \
  --source-root /root/autodl-tmp/TTIE/T073B/source \
  --checkpoint /root/autodl-tmp/TTIE/T073B/shared/epoch=80.ckpt \
  --low-dir /root/autodl-tmp/TTIE/T072AZ/input/T072I_input_cache \
  --low-receipt /root/autodl-tmp/TTIE/T073C/shared/T072I_download_receipt.json \
  --out /root/TTIE_T073B_static_full \
  > /root/autodl-tmp/TTIE/T073B/runs/static_full.log 2>&1
status=$?
printf '%s\n' "$status" > /root/autodl-tmp/TTIE/T073B/runs/static_full.exit
