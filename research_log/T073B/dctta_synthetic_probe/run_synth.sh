#!/usr/bin/env bash
# Synthetic-only native-size DCTTA probe; waits for the Ours-TTT exit receipt before using the GPU.
S=/root/TTIE_T073B_dctta_synth
C=/root/autodl-tmp/TTIE/T073B/runs/dctta_cpu_tests
P=/root/miniconda3/bin/python
while [ ! -f /root/autodl-tmp/TTIE/T073C/recovery/runs/uhdll_ttt_abstain_full.exit ]; do sleep 30; done
nvidia-smi --query-compute-apps=pid,used_memory --format=csv > $S/gpu_before.txt
COMMON="--source-root /root/autodl-tmp/TTIE/T073B/source --checkpoint /root/autodl-tmp/TTIE/T073B/shared/epoch=80.ckpt --low-dir $S/low --low-receipt $S/receipt.json"
cd $S/work || exit 1
for spec in "order16 --order-only" "order0 --order-only --num-workers 0" "full_a --expected-order $S/order16/adaptation_order.json" "full_b --expected-order $S/order16/adaptation_order.json"; do
  set -- $spec; name=$1; shift
  $P -B $C/probe_dctta_native_synthetic.py 3 $COMMON --out $S/$name "$@" > $S/$name.log 2>&1
  echo "$name $?" >> $S/status.txt
done
echo done >> $S/status.txt
