#!/bin/bash
# Poll until every v2 row of a target is frozen: each row runs as soon as its source row's freeze receipt exists.
# usage: DS=SMID [WORKERS=4 STRIDE=5 REPRO=10] watch_v2.sh     (stops when all 10 rows are frozen or after MAX_H hours)
set -u
DS=${DS:?set DS}
V=/root/autodl-tmp/TTIE/T075A/codev2/research_log/T075A/v2
FZ=/root/autodl-tmp/TTIE/T075A/freeze/$DS
CPU_ROWS="ours_v2 retinexformer_plus_D snr_aware_plus_D promptir_plus_D promptir_dctta_plus_D mr_illuminate_plus_D quadprior_plus_D ours_step0_plus_D"
end=$(( $(date +%s) + ${MAX_H:-24} * 3600 ))
while [ $(date +%s) -lt $end ]; do
  bash $V/run_v2.sh ours_ttt_sdsd_knobs ours_v2_sdsd_knobs $CPU_ROWS
  n=$(ls -d $FZ/*/freeze_receipt.json 2>/dev/null | wc -l)
  [ "$n" -ge 10 ] && { echo "ALL 10 FROZEN"; exit 0; }
  sleep 300
done
echo "TIMEOUT"
