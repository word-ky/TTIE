#!/usr/bin/env bash
set -euo pipefail
export PATH=/home/wenchang/asdasdsad/wjq/TTIE/.venv/bin:$PATH
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m ttie.soft_basis.prepare --source-sha "$1" \
  --audit /media/wenchang/F/wjq/TTIE/runs/20260912-213014-ttie-t015-fresh-ready/artifacts/audit \
  --episode-list research_log/T016B_existing_episode_paths.txt \
  --output "$AUTODL_ARTIFACTS_DIR/label_free_inputs"
python -m ttie.soft_basis.score --source-sha "$1" \
  --inputs "$AUTODL_ARTIFACTS_DIR/label_free_inputs" \
  --source-manifest research_log/T014_source_manifest.json \
  --model-identity /home/wenchang/asdasdsad/wjq/TTIE/shared/t004/model_identity.json \
  --prototypes research_log/T016B_frozen_prototypes.pt \
  --receipt research_log/T007_joint_calibration.json \
  --energy research_log/T014_energy.pt --control research_log/T014_value_only.pt \
  --energy-receipt research_log/T014_energy_receipt.json \
  --output "$AUTODL_ARTIFACTS_DIR/scoring" --device cuda:0
python -m ttie.soft_basis.evaluate --source-sha "$1" \
  --selection-dir "$AUTODL_ARTIFACTS_DIR/scoring" \
  --reference-table /home/wenchang/asdasdsad/wjq/TTIE/runs/20260913-000502-ttie-t016a-screen/artifacts/audit/candidate_metrics.json \
  --output "$AUTODL_ARTIFACTS_DIR/evaluation"
