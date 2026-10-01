#!/bin/bash
# T075-A v2 row pipeline (GT-free). usage: DS=<SDSD_indoor|LSRW|SMID> run_v2.sh row1 row2 ...
# rows: ours_ttt_sdsd_knobs (GPU, flock) | ours_v2 | ours_v2_sdsd_knobs | <prereg row>_plus_D
# Each row: produce -> independent verifier -> freeze receipt. A failed row is reported and skipped.
set -u
DS=${DS:?set DS}
P=/root/miniconda3/bin/python
V=/root/autodl-tmp/TTIE/T075A/codev2/research_log/T075A/v2
OUT=/root/autodl-tmp/TTIE/T075A/rows/$DS
FZ=/root/autodl-tmp/TTIE/T075A/freeze/$DS
LG=/root/autodl-tmp/TTIE/T075A/logs/v2_$DS
OURS_SRC=/root/autodl-tmp/TTIE/T073C/recovery/source
OURS_MANIFEST=/root/autodl-tmp/TTIE/T073C/recovery/artifacts/T073C_execution_manifest.json
CLEAN="env -u CUBLAS_WORKSPACE_CONFIG -u OMP_NUM_THREADS -u MKL_NUM_THREADS -u OPENBLAS_NUM_THREADS -u MKL_THREADING_LAYER"
CPU="env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1"
WORKERS=${WORKERS:-4}
STRIDE=${STRIDE:-1}
L=/root/autodl-tmp/TTIE/T074B/targets/$DS
if [ "$DS" = SDSD_indoor ]; then
  SRCF=/root/autodl-tmp/TTIE/T075A/codev2/research_log/T074C/SDSD_indoor   # committed T074-C freeze receipts
  PREFIX=research_log/T075A/sdsd_v2
else
  SRCF=/root/autodl-tmp/TTIE/T075B/freeze/$DS                               # T075-B freeze receipts
  PREFIX=research_log/T075B/$DS
fi
LR=$L/low_receipt.json
N=$($P -c "import json;print(json.load(open('$LR'))['count'])")
mkdir -p $OUT $FZ $LG
st() { echo "$(date -u +%FT%TZ) $DS $*" | tee -a $LG/status.log; }
freeze_of() {  # row -> its freeze receipt (prereg rows from SRCF, v2 rows from FZ)
  case $1 in ours_ttt_sdsd_knobs|ours_v2|ours_v2_sdsd_knobs|*_plus_D) echo $FZ/$1/freeze_receipt.json ;;
  *) echo $SRCF/$1/freeze_receipt.json ;; esac; }
root_of() { $P -c "import json;print(json.load(open('$1'))['remote_output_root'])"; }

plus_d() {  # row source
  local row=$1 src=$2 sf; sf=$(freeze_of $src)
  test -s $sf || { st "SKIP $row: source $src not frozen"; return 1; }
  local sd; sd=$(root_of $sf)
  $CPU $P -B $V/denoise_d.py --source-dir $sd --source-row $src --source-freeze $sf --method-id $row \
    --low-receipt $LR --expected-count $N --workers $WORKERS --out $OUT/$row > $LG/$row.log 2>&1 || { st "FAIL produce $row"; return 2; }
  $CPU $P -B $V/verify_v2_rows.py --kind plus_d --method-id $row --low-receipt $LR --outputs $OUT/$row \
    --expected-count $N --source-dir $sd --source-row $src --recompute-stride $STRIDE --out $OUT/$row.verify.json \
    > $LG/$row.verify.log 2>&1 || { st "FAIL verify $row"; return 3; }
  $P -B $V/make_freeze_v2.py --target $DS --row $row --outputs $OUT/$row --source-freeze $sf --out $FZ/$row \
    --path-prefix $PREFIX/$row > $LG/$row.freeze.log 2>&1 || { st "FAIL freeze $row"; return 4; }
  st "FROZEN $row $(tail -1 $LG/$row.freeze.log)"
}

knobs() {
  local sf; sf=$(freeze_of ours_ttt)
  test -s $sf || { st "SKIP ours_ttt_sdsd_knobs: ours_ttt not frozen"; return 1; }
  local sd; sd=$(root_of $sf)
  local extra=""
  [ "$DS" = SDSD_indoor ] && extra="--expect-manifest /root/autodl-tmp/TTIE/T074C/tuning/SDSD_indoor/ours_ttt_target_tuned/output_manifest.json"
  (cd $OURS_SRC && flock /root/autodl-tmp/TTIE/gpu.lock $CLEAN $P -B $V/run_ours_knobs.py --manifest $OURS_MANIFEST \
    --low-receipt $LR --low-dir $L/low --expected-count $N --frozen-ours-ttt $sd --repro-count ${REPRO:-10} $extra \
    --out $OUT/ours_ttt_sdsd_knobs) > $LG/ours_ttt_sdsd_knobs.log 2>&1 || { st "FAIL produce ours_ttt_sdsd_knobs"; return 2; }
  $CPU $P -B $V/verify_v2_rows.py --kind knobs --method-id ours_ttt_sdsd_knobs --low-receipt $LR \
    --outputs $OUT/ours_ttt_sdsd_knobs --expected-count $N --frozen-ours-ttt $sd \
    --out $OUT/ours_ttt_sdsd_knobs.verify.json > $LG/ours_ttt_sdsd_knobs.verify.log 2>&1 || { st "FAIL verify knobs"; return 3; }
  $P -B $V/make_freeze_v2.py --target $DS --row ours_ttt_sdsd_knobs --outputs $OUT/ours_ttt_sdsd_knobs --source-freeze $sf \
    --out $FZ/ours_ttt_sdsd_knobs --path-prefix $PREFIX/ours_ttt_sdsd_knobs > $LG/ours_ttt_sdsd_knobs.freeze.log 2>&1 \
    || { st "FAIL freeze knobs"; return 4; }
  st "FROZEN ours_ttt_sdsd_knobs $(tail -1 $LG/ours_ttt_sdsd_knobs.freeze.log)"
}

for row in "$@"; do
  test -e $FZ/$row/freeze_receipt.json && { st "ALREADY $row"; continue; }
  case $row in
    ours_ttt_sdsd_knobs) knobs ;;
    ours_v2) plus_d ours_v2 ours_ttt ;;
    ours_v2_sdsd_knobs) plus_d ours_v2_sdsd_knobs ours_ttt_sdsd_knobs ;;
    *_plus_D) plus_d $row ${row%_plus_D} ;;
    *) st "UNKNOWN $row" ;;
  esac
done
st "DONE $*"
