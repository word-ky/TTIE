#!/bin/bash
set -u
root=/root/autodl-tmp/TTIE/T073C/recovery
cd "$root/source" || exit 1
echo "$(date -u +%FT%TZ) START" > "$root/runs/uhdll_ttt_full.log"
/root/miniconda3/bin/python -B -m research_log.T073C.run_ours_ttt_batch \
  --manifest "$root/artifacts/T073C_execution_manifest.json" \
  --low-receipt /root/autodl-tmp/TTIE/T073C/shared/T072I_download_receipt.json \
  --low-dir /root/autodl-tmp/TTIE/T072AZ/input/T072I_input_cache \
  --out "$root/runs/uhdll_ttt_full" >> "$root/runs/uhdll_ttt_full.log" 2>&1
rc=$?
echo "$(date -u +%FT%TZ) EXIT $rc" >> "$root/runs/uhdll_ttt_full.log"
echo "$rc" > "$root/runs/uhdll_ttt_full.exit"
exit "$rc"
