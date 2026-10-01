#!/usr/bin/env bash
set -euo pipefail

pid=86147
while kill -0 "$pid" 2>/dev/null; do sleep 30; done

cd /root/wjq/TTT-ImageEnhancement/experiments/T073C/source
py=/root/wjq/CMA/external_baselines/segllm-venv/bin/python
base=/root/wjq/TTT-ImageEnhancement/experiments/T075A/codev2/research_log/T075A/v2/ceiling.py
work=/root/wjq/TTT-ImageEnhancement/checkpoints/LSRW/tuning_g4_round3_lowq_v2
manifest=/root/wjq/TTT-ImageEnhancement/T073C/recovery/artifacts/T073C_execution_manifest_G4_compat.json
frozen=/root/wjq/TTT-ImageEnhancement/T073C/recovery/artifacts/T073C_execution_manifest_A6000.json
ceil=/root/wjq/TTT-ImageEnhancement/checkpoints/LSRW/ceiling_g4_round3_lowq

common=(
  --cross-environment-exploration
  --target LSRW
  --low-receipt /root/wjq/TTT-ImageEnhancement/T074B/targets/LSRW/low_receipt.json
  --low-dir /root/wjq/TTT-ImageEnhancement/T074B/targets/LSRW/low
  --opaque-manifest /root/wjq/TTT-ImageEnhancement/T074B/targets/LSRW/reference_opaque_manifest.json
  --gate-receipt /root/wjq/TTT-ImageEnhancement/experiments/T075A/gatetree/research_log/T075B/LSRW/reference_gate_receipt.json
  --rows-dir /root/wjq/TTT-ImageEnhancement/experiments/T075A/gatetree/research_log/T075B/LSRW
  --stage-target /root/wjq/TTT-ImageEnhancement/T074B/stage_target.py
  --manifest "$manifest"
  --frozen-manifest "$frozen"
)

export CUDA_VISIBLE_DEVICES=2
"$py" -u "$base" tuning "${common[@]}" --round2 --work "$work"
"$py" -u "$base" tuning "${common[@]}" --materialize --work "$work"

unset CUDA_VISIBLE_DEVICES
"$py" -u "$base" kappa "${common[@]}" --tuned-row "$work/ours_ttt_target_tuned" --out "$ceil/row" --workers 4
"$py" -u "$base" metrics "${common[@]}" --tuned-row "$work/ours_ttt_target_tuned" --ceiling "$ceil/row" --out "$ceil/metrics" --workers 4 --reuse-scores /root/wjq/TTT-ImageEnhancement/checkpoints/LSRW/frozen_metrics_v2_result.json
echo PIPELINE_EXIT=0
