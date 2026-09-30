#!/usr/bin/env bash
set -euo pipefail

cd /root/wjq/TTT-ImageEnhancement/experiments/T073C/source
py=/root/wjq/CMA/external_baselines/segllm-venv/bin/python
base=/root/wjq/TTT-ImageEnhancement/experiments/T075A/codev2/research_log/T075A/v2/targets_sid.py
work=/root/wjq/TTT-ImageEnhancement/checkpoints/SID/tuning_g4
manifest=/root/wjq/TTT-ImageEnhancement/T073C/recovery/artifacts/T073C_execution_manifest_G4_compat.json
frozen=/root/wjq/TTT-ImageEnhancement/T073C/recovery/artifacts/T073C_execution_manifest_A6000.json
common=(
  --target SID
  --low-receipt /root/wjq/TTT-ImageEnhancement/T074B/targets/SID/low_receipt.json
  --opaque-manifest /root/wjq/TTT-ImageEnhancement/T074B/targets/SID/reference_opaque_manifest.json
  --gate-receipt /root/wjq/TTT-ImageEnhancement/experiments/T075A/gatetree/research_log/T075B/SID/reference_gate_receipt.json
  --rows-dir /root/wjq/TTT-ImageEnhancement/experiments/T075A/gatetree/research_log/T075B/SID
  --stage-target /root/wjq/TTT-ImageEnhancement/T074B/stage_target.py
)

export CUDA_VISIBLE_DEVICES=0
"$py" -u "$base" ceiling tuning "${common[@]}" \
  --low-dir /root/wjq/TTT-ImageEnhancement/T074B/targets/SID/low \
  --manifest "$manifest" --frozen-manifest "$frozen" --round2 --work "$work"
"$py" -u "$base" ceiling tuning "${common[@]}" \
  --low-dir /root/wjq/TTT-ImageEnhancement/T074B/targets/SID/low \
  --manifest "$manifest" --frozen-manifest "$frozen" --materialize --work "$work"

unset CUDA_VISIBLE_DEVICES
ceil=/root/wjq/TTT-ImageEnhancement/checkpoints/SID/ceiling_g4
"$py" -u "$base" ceiling kappa "${common[@]}" \
  --tuned-row "$work/ours_ttt_target_tuned" --out "$ceil/row" --workers 4
"$py" -u "$base" ceiling metrics "${common[@]}" \
  --tuned-row "$work/ours_ttt_target_tuned" --ceiling "$ceil/row" --out "$ceil/metrics" --workers 4
echo SID_PIPELINE_EXIT=0
