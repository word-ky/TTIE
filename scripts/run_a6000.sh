#!/usr/bin/env bash
set -euo pipefail
export PATH=/home/wenchang/asdasdsad/wjq/TTIE/.venv/bin:$PATH
mkdir -p "$AUTODL_ARTIFACTS_DIR"
python -c 'import sys,torch,PIL; print(sys.version); print("torch",torch.__version__,"CUDA",torch.version.cuda,"Pillow",PIL.__version__); print(torch.cuda.get_device_name(0))' | tee "$AUTODL_ARTIFACTS_DIR/environment.txt"
python -m unittest discover -s tests -v 2>&1 | tee "$AUTODL_ARTIFACTS_DIR/tests.txt"
python -m ttie.demo --device cpu --output "$AUTODL_ARTIFACTS_DIR/cpu"
python -m ttie.demo --device cuda:0 --output "$AUTODL_ARTIFACTS_DIR/cuda"
python -m ttie.demo --device cuda:0 --output "$AUTODL_ARTIFACTS_DIR/cuda_repeat"
python - <<'PY' | tee "$AUTODL_ARTIFACTS_DIR/reproducibility.txt"
import json
import os
from pathlib import Path
import torch
p = Path(os.environ['AUTODL_ARTIFACTS_DIR'])
first = json.loads((p / 'cuda/metrics.json').read_text())
repeat = json.loads((p / 'cuda_repeat/metrics.json').read_text())
assert first == repeat
a = torch.load(p / 'cuda/tensors.pt', weights_only=True)
b = torch.load(p / 'cuda_repeat/tensors.pt', weights_only=True)
for key in a:
    torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
cpu = json.loads((p / 'cpu/metrics.json').read_text())
for mode in ('identity', 'global', 'spatial'):
    assert first['cases'][mode]['all_finite']
    print(mode, 'CPU MSE', cpu['cases'][mode]['mse'], 'CUDA MSE', first['cases'][mode]['mse'])
print('CUDA repeated reports and tensors match exactly; all finite.')
PY
