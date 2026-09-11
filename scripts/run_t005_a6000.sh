#!/usr/bin/env bash
set -euo pipefail
export PATH=/home/wenchang/asdasdsad/wjq/TTIE/.venv/bin:$PATH
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$AUTODL_ARTIFACTS_DIR"
python - <<'PY' | tee "$AUTODL_ARTIFACTS_DIR/environment.txt"
import sys, torch, importlib.metadata as m
print(sys.version)
for package in ['torch','torchvision','open_clip_torch','timm','Pillow']:
    print(package, m.version(package))
print('CUDA',torch.version.cuda,torch.cuda.get_device_name(0))
PY
python -m unittest discover -s tests -v 2>&1 | tee "$AUTODL_ARTIFACTS_DIR/tests.txt"
python -m ttie.relative_audit \
  --manifest research_log/T005_manifest.json \
  --images /home/wenchang/asdasdsad/wjq/TTIE/shared/t005/images \
  --model-identity /home/wenchang/asdasdsad/wjq/TTIE/shared/t004/model_identity.json \
  --absolute-calibration /home/wenchang/asdasdsad/wjq/TTIE/runs/20260912-045537-ttie-t004-a6000/artifacts/audit/calibration.json \
  --output "$AUTODL_ARTIFACTS_DIR/audit" --device cuda:0
