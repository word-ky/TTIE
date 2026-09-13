import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from ttie.ssim_audit import select_rows
from ttie.ssim_transfer import rgb_ssim

ROOT = Path(__file__).resolve().parents[1]


def test_accepted_primary_selection():
    rows, clean = select_rows(json.loads((ROOT/'research_log/T021A_outputs_verified.json').read_bytes()))
    assert len(rows) == 200 and len(clean) == 40
    assert not any('offset' in r['condition'] for r in rows)


def test_independent_kernel_including_borders():
    spec = importlib.util.spec_from_file_location('independent', ROOT/'research_log/T021A_independent_replay.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    x = np.random.default_rng(17).random((27,31,3)).astype(np.float32)
    y = x.copy(); y[:5] *= .7; y[-5:] *= .8
    tensor = lambda a: torch.from_numpy(a).permute(2,0,1).unsqueeze(0)
    assert abs(module.ssim(tensor(x),tensor(y))-rgb_ssim(x,y)) < 1e-12
