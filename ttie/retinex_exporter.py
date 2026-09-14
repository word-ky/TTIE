"""Retinexformer default_no_gt_mean, native RGB float export.

Calls the unmodified official network from a pinned external source tree.
Forward conventions: caiyuanhao1998/Retinexformer commit
1e9a0efce4b306b6701b824768370ff26066c32a, Enhancement/test_from_dataset.py.
MIT Copyright (c) 2023 Yuanhao Cai; notice in research_log/T027A_official/LICENSE.txt.
This exporter is bound to the native LOL-v2 Real geometry (600 x 400).
"""
import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_model(checkpoint, config):
    config = json.loads(Path(config).read_text())
    spec = importlib.util.spec_from_file_location('retinexformer_export_arch', config['architecture'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.RetinexFormer(**config['network'])
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['params'], strict=True)
    return torch.nn.DataParallel(model.cuda()).eval()


def forward(input_, model):
    h, w = input_.shape[-2:]
    input_ = F.pad(input_, (0, (-w) % 4, 0, (-h) % 4), 'reflect')
    restored = model(input_)[:, :, :h, :w]
    return torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--low', nargs='+', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--out', required=True)
    return p


def main():
    import cv2

    args = parser().parse_args()
    torch.manual_seed(7)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = load_model(args.checkpoint, args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.inference_mode():
        for name in args.low:
            path = Path(name)
            img = np.float32(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)) / 255.
            x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            y = forward(x, model)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            np.save(out / (path.stem + '.npy'), y)
            rows.append({'low': str(path), 'low_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'shape': list(y.shape), 'finite': bool(np.isfinite(y).all()),
                         'output_sha256': hashlib.sha256(y.tobytes()).hexdigest(),
                         'runtime_s': elapsed, 'peak_memory_bytes': torch.cuda.max_memory_allocated()})
            print(path.name, elapsed, flush=True)
    (out / 'receipt.json').write_text(json.dumps({'mode': 'default_no_gt_mean',
        'GT_mean': False, 'self_ensemble': False, 'rows': rows}, indent=2) + '\n')


if __name__ == '__main__':
    main()
