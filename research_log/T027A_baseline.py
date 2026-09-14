"""Eight official forward calls, with an image-read allowlist and no targets."""
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from T027A_official_path import load_official, official_forward

root = Path(sys.argv[1])
manifest = json.loads((root / 'smoke_manifest.json').read_text())
paths = [(root / 'low' / row['low']).resolve() for row in manifest['selected']]
allowed = set(paths)
decoded = []
read = cv2.imread


def low_read(path, *a, **kw):
    p = Path(path).resolve()
    if p not in allowed:
        raise PermissionError('Image outside smoke allowlist: ' + str(p))
    decoded.append(str(p))
    return read(str(p), *a, **kw)


def audit(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes)):
        p = Path(args[0]).resolve()
        if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif'} and p not in allowed:
            raise PermissionError('Image outside smoke allowlist: ' + str(p))


cv2.imread = low_read
sys.addaudithook(audit)
torch.manual_seed(7)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
model = load_official(root / 'official', root / 'LOL_v2_real.pth',
                      root / 'official/Options/RetinexFormer_LOL_v2_real.yml')
forward, source = official_forward(root / 'official')
out = root / 'baseline'
out.mkdir(exist_ok=True)
(out / 'executed_official_forward.py').write_text(source + '\n')
rows = []
with torch.inference_mode():
    for row, p in zip(manifest['selected'], paths):
        img = np.float32(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)) / 255.
        x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        y = forward(x, model)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        assert y.shape == img.shape == (400, 600, 3) and np.isfinite(y).all()
        np.save(out / (p.stem + '.npy'), y)
        rows.append({'low': row['low'], 'shape': list(y.shape),
                     'output_sha256': hashlib.sha256(y.tobytes()).hexdigest(),
                     'runtime_s': elapsed, 'peak_memory_bytes': torch.cuda.max_memory_allocated()})
        print(row['low'], elapsed, flush=True)
receipt = {'path': 'unchanged official forward AST; direct official network class and strict checkpoint load',
           'GT_mean': False, 'self_ensemble': False, 'decoded_paths': decoded,
           'target_access': 'all non-allowlisted image opens denied; no target argument or load',
           'torch': torch.__version__, 'cuda': torch.version.cuda,
           'gpu': torch.cuda.get_device_name(), 'rows': rows}
(out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print('BASELINE PASS:8 low-only official forwards', flush=True)
