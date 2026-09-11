#!/usr/bin/env bash
set -euo pipefail
export PATH=/home/wenchang/asdasdsad/wjq/TTIE/.venv/bin:$PATH
mkdir -p "$AUTODL_ARTIFACTS_DIR"
python -c 'import sys,torch,PIL; print(sys.version); print("torch",torch.__version__,"CUDA",torch.version.cuda,"Pillow",PIL.__version__); print(torch.cuda.get_device_name(0))' | tee "$AUTODL_ARTIFACTS_DIR/environment.txt"
python -m unittest discover -s tests -v 2>&1 | tee "$AUTODL_ARTIFACTS_DIR/tests.txt"
python - <<'PY' | tee "$AUTODL_ARTIFACTS_DIR/parity.txt"
import json
import torch
from ttie.demo import run_demo
from ttie.suite import case_results
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
baseline = json.load(open('research_log/artifacts/T001_cpu/metrics.json'))
actual, _ = run_demo()
for mode in ('identity', 'global', 'spatial'):
    assert abs(actual['cases'][mode]['mse'] - baseline['cases'][mode]['mse']) < 1e-9
print('T001 frozen numeric regression passed (MSE absolute tolerance 1e-9).')
args = dict(family='midtone', condition='left_right', seed=7, steps=200, lr=0.03)
cpu_rows, cpu_pack, _ = case_results(device='cpu', **args)
gpu_rows, gpu_pack, gpu_diag = case_results(device='cuda:0', **args)
repeat_rows, repeat_pack, repeat_diag = case_results(device='cuda:0', **args)
assert gpu_rows == repeat_rows and gpu_diag == repeat_diag
for key in gpu_pack:
    torch.testing.assert_close(gpu_pack[key], repeat_pack[key], rtol=0, atol=0)
    torch.testing.assert_close(cpu_pack[key], gpu_pack[key], rtol=1e-4, atol=1e-5)
print('All seven variants: CPU/CUDA parity atol=1e-5 rtol=1e-4; CUDA repeat exact.')
PY
python -m ttie.suite --device cuda:0 --output "$AUTODL_ARTIFACTS_DIR/suite"
python -m ttie.summarize "$AUTODL_ARTIFACTS_DIR/suite"
