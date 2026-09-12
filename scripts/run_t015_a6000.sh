#!/usr/bin/env bash
set -euo pipefail
export PATH=/home/wenchang/asdasdsad/wjq/TTIE/.venv/bin:$PATH
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# The Python entry point checks Git provenance before model/data/output work.
# Run the local test suite separately; this wrapper creates no pre-guard artifacts.
exec python -m ttie.routing.pilot --source-sha "$1" \
  --manifest research_log/T015_manifest.json --source-manifest research_log/T014_source_manifest.json \
  --images /home/wenchang/asdasdsad/wjq/TTIE/shared/t008/val2017 \
  --model-identity /home/wenchang/asdasdsad/wjq/TTIE/shared/t004/model_identity.json \
  --prototypes research_log/remote_runs/20260912-071126-ttie-t006-a6000/artifacts/audit/prototypes.pt \
  --receipt research_log/T007_joint_calibration.json \
  --energy research_log/T014_energy.pt --control research_log/T014_value_only.pt \
  --energy-receipt research_log/T014_energy_receipt.json \
  --t006-images /home/wenchang/asdasdsad/wjq/TTIE/shared/t006/images \
  --output "$AUTODL_ARTIFACTS_DIR/audit" --device cuda:0
