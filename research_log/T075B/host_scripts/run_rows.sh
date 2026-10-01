#!/bin/bash
# T075-B non-tuned row pipeline: per row smoke -> smoke check -> (disk check) -> full -> independent verifier.
# usage: DS=<LSRW|SMID> run_rows.sh row1 row2 ...   Every GPU step is serialized with flock on gpu.lock.
# A failed row is recorded and skipped (no rescue); ours_ttt is skipped if ours_step0 did not verify.
set -u
. /root/autodl-tmp/TTIE/T075B/vars.sh
LOCK=/root/autodl-tmp/TTIE/gpu.lock
LG=$TC/logs/$DS
mkdir -p $R $LG
ST=$LG/status.log
st() { echo "$(date -u +%FT%TZ) $*" | tee -a $ST; }
g() { flock $LOCK "$@"; }          # GPU step, serialized
smoke_check() { $P -B $TC/code/check_smoke.py $L/low_receipt.json $1 > $1.smokecheck.json 2>&1; }
disk_ok() {  # $1 = smoke dir: per-image subdirs x N (+10%) + top-level files must leave >= 5 GiB free
  local per=$(find $1 -mindepth 1 -maxdepth 1 -type d -exec du -sb {} + | awk '{s+=$1} END {print s+0}')
  local const=$(find $1 -maxdepth 1 -type f -printf '%s\n' | awk '{s+=$1} END {print s+0}')
  local free=$(df -B1 --output=avail /root/autodl-tmp | tail -1)
  local need=$(( per * N * 11 / 10 + const ))
  st "DISK $1 per_image_bytes=$per const=$const projected_full=$need free=$free"
  [ $(( free - need )) -ge $(( 5 * 1024 * 1024 * 1024 )) ]
}
vgen() {  # kind dir extra...
  local k=$1 d=$2; shift 2
  $P -B $C/verify_generic_outputs.py --kind $k --low-receipt $L/low_receipt.json --outputs $R/$d \
    --expected-count $N --out $R/$d.verify.json "$@" > $LG/$d.verify.log 2>&1
}
first_row() { $P -c "import json;print(json.load(open('$1/output_manifest.json'))['rows'][0]['output_tensor_sha256'])"; }
msha() { sha256sum $1/output_manifest.json | cut -d' ' -f1; }

run_row() {
  local row=$1
  case $row in
  ours_step0)
    cd $OURS_SRC
    g $CLEAN $P -B $C/run_ours_step0_generic.py --manifest $OURS_MANIFEST --low-receipt $L/low_receipt.json \
      --low-dir $L/low --expected-count $N --smoke-one --out $R/ours_step0_smoke > $LG/ours_step0_smoke.log 2>&1 || return 11
    smoke_check $R/ours_step0_smoke || return 12; disk_ok $R/ours_step0_smoke || return 99
    g $CLEAN $P -B $C/run_ours_step0_generic.py --manifest $OURS_MANIFEST --low-receipt $L/low_receipt.json \
      --low-dir $L/low --expected-count $N --out $R/ours_step0 > $LG/ours_step0.log 2>&1 || return 21
    vgen ours_step0 ours_step0 --execution-manifest $OURS_MANIFEST || return 31 ;;
  ours_ttt)
    test -s $R/ours_step0.verify.json || return 1
    local H=$(msha $R/ours_step0); cd $OURS_SRC
    g $CLEAN $P -B $C/run_ours_ttt_abstain_generic.py --manifest $OURS_MANIFEST --low-receipt $L/low_receipt.json \
      --low-dir $L/low --step0-manifest $R/ours_step0/output_manifest.json --step0-root $R/ours_step0 \
      --expected-step0-manifest-sha256 $H --smoke-count 1 --out $R/ours_ttt_smoke > $LG/ours_ttt_smoke.log 2>&1 || return 11
    smoke_check $R/ours_ttt_smoke || return 12; disk_ok $R/ours_ttt_smoke || return 99
    g $CLEAN $P -B $C/run_ours_ttt_abstain_generic.py --manifest $OURS_MANIFEST --low-receipt $L/low_receipt.json \
      --low-dir $L/low --step0-manifest $R/ours_step0/output_manifest.json --step0-root $R/ours_step0 \
      --expected-step0-manifest-sha256 $H --out $R/ours_ttt > $LG/ours_ttt.log 2>&1 || return 21
    vgen ours_ttt ours_ttt --step0-outputs $R/ours_step0 --execution-manifest $OURS_MANIFEST || return 31 ;;
  retinexformer|snr_aware)
    cd $AZ/runtime
    g $AZ_ENV $P -B $C/run_retinex_snr_generic.py --method $row --runtime-root $AZ/runtime \
      --bindings $AZ/T072AZ/seal/baseline_bindings.json --low-dir $L/low --low-receipt $L/low_receipt.json \
      --expected-count $N --smoke-count 1 --out $R/${row}_smoke > $LG/${row}_smoke.log 2>&1 || return 11
    smoke_check $R/${row}_smoke || return 12; disk_ok $R/${row}_smoke || return 99
    g $AZ_ENV $P -B $C/run_retinex_snr_generic.py --method $row --runtime-root $AZ/runtime \
      --bindings $AZ/T072AZ/seal/baseline_bindings.json --low-dir $L/low --low-receipt $L/low_receipt.json \
      --expected-count $N --out $R/$row > $LG/$row.log 2>&1 || return 21
    vgen $row $row || return 31 ;;
  promptir)
    cd $B73/shared
    g $CLEAN T073B_SHARED=$B73/shared $P -B $C/run_promptir_static_generic.py --source-root $B73/source \
      --checkpoint $B73/shared/epoch=80.ckpt --low-dir $L/low --low-receipt $L/low_receipt.json \
      --expected-count $N --smoke-one --out $R/promptir_smoke > $LG/promptir_smoke.log 2>&1 || return 11
    smoke_check $R/promptir_smoke || return 12; disk_ok $R/promptir_smoke || return 99
    g $CLEAN T073B_SHARED=$B73/shared $P -B $C/run_promptir_static_generic.py --source-root $B73/source \
      --checkpoint $B73/shared/epoch=80.ckpt --low-dir $L/low --low-receipt $L/low_receipt.json \
      --expected-count $N --out $R/promptir > $LG/promptir.log 2>&1 || return 21
    vgen promptir promptir || return 31 ;;
  promptir_dctta)
    for w in dctta_smoke_work dctta_order_work dctta_full_work; do
      mkdir -p $R/$w && cp $B73/source/pretrain/wavelet.mat $R/$w/; done
    local DA="$CLEAN T073B_SHARED=$B73/shared $P -B $C/run_dctta_generic.py --source-root $B73/source --checkpoint $B73/shared/epoch=80.ckpt --low-dir $L/low --low-receipt $L/low_receipt.json --expected-count $N"
    cd $R/dctta_smoke_work; g $DA --smoke-one --out $R/dctta_smoke > $LG/dctta_smoke.log 2>&1 || return 11
    smoke_check $R/dctta_smoke || return 12; disk_ok $R/dctta_smoke || return 99
    cd $R/dctta_order_work; g $DA --order-only --out $R/dctta_order > $LG/dctta_order.log 2>&1 || return 15
    st "DCTTA_ORDER_SHA256 $(sha256sum $R/dctta_order/adaptation_order.json | cut -d' ' -f1)"
    cd $R/dctta_full_work; g $DA --expected-order $R/dctta_order/adaptation_order.json --out $R/dctta > $LG/dctta.log 2>&1 || return 21
    vgen promptir_dctta dctta || return 31 ;;
  quadprior)
    cd $TC
    g $QP -B $QC/run_quadprior_batch.py --source-root $QS $QW --low-dir $L/low --low-receipt $L/low_receipt.json \
      --expected-count $N --smoke-one --out $R/quadprior_smoke --workdir $R/quadprior_smoke_cwd > $LG/quadprior_smoke.log 2>&1 || return 11
    smoke_check $R/quadprior_smoke || return 12; disk_ok $R/quadprior_smoke || return 99
    g $QP -B $QC/run_quadprior_batch.py --source-root $QS $QW $QPIN --low-dir $L/low --low-receipt $L/low_receipt.json \
      --expected-count $N --out $R/quadprior --workdir $R/quadprior_cwd > $LG/quadprior.log 2>&1 || return 21
    $P -B $QC/verify_quadprior_full.py --low-receipt $L/low_receipt.json --low-dir $L/low --outputs $R/quadprior \
      --out $R/quadprior.verify.json --expected-count $N $QPIN --expected-output-manifest-sha256 $(msha $R/quadprior) \
      --first-row-tensor-sha256 $(first_row $R/quadprior_smoke) --require-promotable > $LG/quadprior.verify.log 2>&1 || return 31 ;;
  mr_illuminate)
    cd $TC
    g $MENV $MP -B $MC/run_mri_native_batch.py --source-root $MS --vae-checkpoint $WT/main-epoch=00-step=7000.ckpt \
      --sd-snapshot $SNAP --low-dir $L/low --low-receipt $L/low_receipt.json --expected-count $N --smoke-one \
      --out $R/mri_smoke > $LG/mri_smoke.log 2>&1 || return 11
    smoke_check $R/mri_smoke || return 12; disk_ok $R/mri_smoke || return 99
    g $MENV $MP -B $MC/run_mri_native_batch.py --source-root $MS --vae-checkpoint $WT/main-epoch=00-step=7000.ckpt \
      --sd-snapshot $SNAP --low-dir $L/low --low-receipt $L/low_receipt.json --expected-count $N $MPIN \
      --out $R/mri > $LG/mri.log 2>&1 || return 21
    $P -B $MC/verify_mri_full.py --low-receipt $L/low_receipt.json --low-dir $L/low --outputs $R/mri \
      --out $R/mri.verify.json --expected-count $N $MPIN --expected-output-manifest-sha256 $(msha $R/mri) \
      --first-row-tensor-sha256 $(first_row $R/mri_smoke) --require-promotable > $LG/mri.verify.log 2>&1 || return 31 ;;
  *) return 2 ;;
  esac
}

st "START DS=$DS N=$N rows=$*"
for row in "$@"; do
  st "ROW_BEGIN $row"
  run_row $row; rc=$?
  if [ $rc -eq 0 ]; then st "ROW_VERIFIED $row"
  elif [ $rc -eq 99 ]; then st "ROW_STOP_DISK $row (projected output would leave < 5 GiB); pipeline stopped"; break
  else st "ROW_FAILED $row rc=$rc (11 smoke run, 12 smoke check, 15 order-only, 21 full run, 31 verifier, 1 dependency)"; fi
done
st "PIPELINE_END"
