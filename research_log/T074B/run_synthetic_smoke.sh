#!/bin/bash
# T074-B synthetic execution smoke of every generic runner (synthetic inputs only).
set -u
P=/root/miniconda3/bin/python
T=/root/autodl-tmp/TTIE/T074B
C=$T/code
R=${R:-$T/runs/synthetic_smoke}
OURS_SRC=/root/autodl-tmp/TTIE/T073C/recovery/source
OURS_MANIFEST=/root/autodl-tmp/TTIE/T073C/recovery/artifacts/T073C_execution_manifest.json
B73=/root/autodl-tmp/TTIE/T073B
AZ=/root/autodl-tmp/TTIE/T072AZ
# Same process environment as the frozen UHD-LL executions: Ours and PromptIR/DCTTA ran with these
# unset (FinalOurs asserts it); the T072-AZ RetinexFormer/SNR processes ran with them set.
unset CUBLAS_WORKSPACE_CONFIG OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS MKL_THREADING_LAYER
AZ_ENV="env CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1"
export T073B_SHARED=$B73/shared
mkdir -p "$R"
status() { echo "$(date -u +%FT%TZ) $1 rc=$2" | tee -a "$R/status.log"; }
verify() { $P -B $C/verify_generic_outputs.py --kind "$1" --low-receipt $T/synthetic/$2/low_receipt.json \
  --outputs "$3" --expected-count 3 --out "$3.verify.json" "${@:4}" > "$3.verify.log" 2>&1; status "verify_$1_$2" $?; }

for S in ${SETS:-A B}; do
  L=$T/synthetic/$S
  cd $OURS_SRC
  $P -B $C/run_ours_step0_generic.py --manifest $OURS_MANIFEST --low-receipt $L/low_receipt.json \
    --low-dir $L/low --expected-count 3 --out $R/step0_$S > $R/step0_$S.log 2>&1
  status step0_$S $?
  verify ours_step0 $S $R/step0_$S --execution-manifest $OURS_MANIFEST
  H=$(sha256sum $R/step0_$S/output_manifest.json | cut -d' ' -f1)
  $P -B $C/run_ours_ttt_abstain_generic.py --manifest $OURS_MANIFEST --low-receipt $L/low_receipt.json \
    --low-dir $L/low --step0-manifest $R/step0_$S/output_manifest.json --step0-root $R/step0_$S \
    --expected-step0-manifest-sha256 $H --out $R/ttt_$S > $R/ttt_$S.log 2>&1
  status ttt_$S $?
  verify ours_ttt $S $R/ttt_$S --step0-outputs $R/step0_$S --execution-manifest $OURS_MANIFEST

  cd $B73/shared
  $P -B $C/run_promptir_static_generic.py --source-root $B73/source --checkpoint $B73/shared/epoch=80.ckpt \
    --low-dir $L/low --low-receipt $L/low_receipt.json --expected-count 3 --out $R/promptir_$S > $R/promptir_$S.log 2>&1
  status promptir_$S $?
  verify promptir $S $R/promptir_$S

  W=$R/dctta_work_$S
  mkdir -p $W && cp $B73/source/pretrain/wavelet.mat $W/ && cd $W
  $P -B $C/run_dctta_generic.py --source-root $B73/source --checkpoint $B73/shared/epoch=80.ckpt \
    --low-dir $L/low --low-receipt $L/low_receipt.json --expected-count 3 --num-workers 2 \
    --order-only --out $R/dctta_order_$S > $R/dctta_order_$S.log 2>&1
  status dctta_order_$S $?
  $P -B $C/run_dctta_generic.py --source-root $B73/source --checkpoint $B73/shared/epoch=80.ckpt \
    --low-dir $L/low --low-receipt $L/low_receipt.json --expected-count 3 --num-workers 2 \
    --expected-order $R/dctta_order_$S/adaptation_order.json --out $R/dctta_$S > $R/dctta_$S.log 2>&1
  status dctta_$S $?
  verify promptir_dctta $S $R/dctta_$S

  cd $AZ/runtime
  for M in retinexformer snr_aware; do
    $AZ_ENV $P -B $C/run_retinex_snr_generic.py --method $M --runtime-root $AZ/runtime \
      --bindings $AZ/T072AZ/seal/baseline_bindings.json --low-dir $L/low --low-receipt $L/low_receipt.json \
      --expected-count 3 --out $R/${M}_$S > $R/${M}_$S.log 2>&1
    status ${M}_$S $?
    verify $M $S $R/${M}_$S
  done
done
status DONE 0
