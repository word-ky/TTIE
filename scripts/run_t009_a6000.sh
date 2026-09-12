#!/usr/bin/env bash
set -euo pipefail
export PATH=/home/wenchang/asdasdsad/wjq/TTIE/.venv/bin:$PATH
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$AUTODL_ARTIFACTS_DIR"
python - <<'PY' | tee "$AUTODL_ARTIFACTS_DIR/environment.txt"
import sys, torch, importlib.metadata as m
print(sys.version)
for package in ['torch','torchvision','open_clip_torch','timm','Pillow']:
    print(package,m.version(package))
print('CUDA',torch.version.cuda,torch.cuda.get_device_name(0))
PY
python -m unittest discover -s tests -v 2>&1 | tee "$AUTODL_ARTIFACTS_DIR/tests.txt"
python -m ttie.geometry_audit \
  --manifest research_log/T009_manifest.json \
  --images /home/wenchang/asdasdsad/wjq/TTIE/shared/t008/val2017 \
  --model-identity /home/wenchang/asdasdsad/wjq/TTIE/shared/t004/model_identity.json \
  --prototypes research_log/remote_runs/20260912-071126-ttie-t006-a6000/artifacts/audit/prototypes.pt \
  --receipt research_log/T007_joint_calibration.json \
  --t006-images /home/wenchang/asdasdsad/wjq/TTIE/shared/t006/images \
  --output "$AUTODL_ARTIFACTS_DIR/audit" --device cuda:0
