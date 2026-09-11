#!/usr/bin/env bash
set -euo pipefail
export PATH=/home/wenchang/asdasdsad/wjq/TTIE/.venv/bin:$PATH
mkdir -p "$AUTODL_ARTIFACTS_DIR"
python -c 'import sys,torch,PIL; print(sys.version); print("torch",torch.__version__,"CUDA",torch.version.cuda,"Pillow",PIL.__version__); print(torch.cuda.get_device_name(0)); print("Full T003 matrix device: CPU; separate CUDA validation follows.")' | tee "$AUTODL_ARTIFACTS_DIR/environment.txt"
python -m unittest discover -s tests -v 2>&1 | tee "$AUTODL_ARTIFACTS_DIR/tests.txt"
python - <<'PY' | tee "$AUTODL_ARTIFACTS_DIR/device_checks.txt"
import json
import os
from pathlib import Path
import torch
from ttie.safety_sweep import safety_case
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
args = dict(family='midtone', condition='left_right', seed=7, steps=200, lr=.03)
cpu, cp, _ = safety_case(device='cpu', **args)
gpu, gp, gd = safety_case(device='cuda:0', **args)
repeat, rp, rd = safety_case(device='cuda:0', **args)
assert gpu == repeat and gd == rd
for key in gp:
    torch.testing.assert_close(gp[key], rp[key], rtol=0, atol=0)
print('All 13 variants: CUDA repeat rows/diagnostics/tensors exact.')
differences = []
for a, b in zip(cpu, gpu):
    item = dict(model=a['model'], cpu_mse=a['mse'], cuda_mse=b['mse'],
                abs_mse_difference=abs(a['mse']-b['mse']),
                max_output_abs_difference=(cp[a['model']]-gp[a['model']]).abs().max().item())
    differences.append(item)
    assert a['all_finite'] and b['all_finite']
    print(item)
Path(os.environ['AUTODL_ARTIFACTS_DIR'], 'device_checks.json').write_text(json.dumps(differences, indent=2))
print('CPU/CUDA sensitivity measured, not hidden with relaxed tolerances; full matrix is uniformly CPU.')
PY
python -m ttie.safety_sweep --device cpu --output "$AUTODL_ARTIFACTS_DIR/sweep"
python -m ttie.safety_summary "$AUTODL_ARTIFACTS_DIR/sweep"
