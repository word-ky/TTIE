"""Low-only SNR-Aware ttie_native_pad16 float exporter.

TTIE-authored adapter. The network is imported from an external official source
tree; no third-party implementation is vendored. Native pad16 is a protocol
adaptation, not reproduction of the official resize-based test4 numbers.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_model(checkpoint, config, source):
    sys.path.insert(0, str(Path(source).resolve()))
    from models.archs.low_light_transformer import low_light_transformer
    config = json.loads(Path(config).read_text())
    model = low_light_transformer(**config['network'])
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    return torch.nn.DataParallel(model.cuda()).eval()


def pad16(x):
    height, width = x.shape[-2:]
    return F.pad(x, (0, (-width) % 16, 0, (-height) % 16), mode='reflect')


def snr_mask(low, blurred_low):
    def luminance(rgb):
        return rgb[:, :1] * .299 + rgb[:, 1:2] * .587 + rgb[:, 2:3] * .114
    signal = luminance(blurred_low)
    ratio = signal / ((luminance(low) - signal).abs() + .0001)
    maximum = ratio.flatten(1).amax(1).reshape(-1, 1, 1, 1)
    return (ratio / (maximum + .0001)).clamp(0, 1).float()


def forward(image, model):
    import cv2
    height, width = image.shape[-2:]
    # Compute the low-derived feature at native resolution before any padding.
    pixels = image.permute(1, 2, 0).numpy() * 255.0
    blurred = cv2.blur(pixels, (5, 5)) * 1.0 / 255.0
    feature = torch.from_numpy(blurred).float().permute(2, 0, 1)
    x = pad16(image.unsqueeze(0)).cuda()
    nf = pad16(feature.unsqueeze(0)).cuda()
    restored = model(x, snr_mask(x, nf))[:, :, :height, :width].clamp(0, 1)
    return restored[0].detach().cpu().permute(1, 2, 0).numpy()


def parameter_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--low', nargs='+', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--out', required=True)
    return p


def main():
    import cv2
    args = parser().parse_args()
    torch.manual_seed(7)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = load_model(args.checkpoint, args.config, args.source)
    before = parameter_hash(model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.inference_mode():
        for name in args.low:
            p = Path(name)
            raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
            image = torch.from_numpy(np.ascontiguousarray(raw[:, :, [2, 1, 0]].transpose(2, 0, 1))).float()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            y = forward(image, model)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            np.save(out / (p.stem + '.npy'), y)
            rows.append({'low': str(p), 'low_sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                         'shape': list(y.shape), 'finite': bool(np.isfinite(y).all()),
                         'min': float(y.min()), 'max': float(y.max()),
                         'output_sha256': hashlib.sha256(y.tobytes()).hexdigest(),
                         'runtime_s': elapsed, 'peak_memory_bytes': torch.cuda.max_memory_allocated()})
            print(p.name, elapsed, flush=True)
    after = parameter_hash(model)
    assert before == after
    (out / 'receipt.json').write_text(json.dumps({'mode': 'ttie_native_pad16',
        'rows': rows, 'parameter_hash_before': before, 'parameter_hash_after': after}, indent=2) + '\n')


if __name__ == '__main__':
    main()
