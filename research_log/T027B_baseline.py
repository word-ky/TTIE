"""Run eight independent native-pad16 forwards with no quality metrics."""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from T027B_access import install
from T027B_adapter import load_adapter, parameter_hash

root = Path(sys.argv[1])
manifest = json.loads((root / 'smoke_manifest.json').read_text())
paths = [(root / 'low' / r['low']).resolve() for r in manifest['selected']]
decoded, denied = install(paths)
torch.manual_seed(7)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
model, forward, read_img_seq = load_adapter(root)
before = parameter_hash(model)
out = root / 'baseline'
out.mkdir(exist_ok=True)
rows = []
with torch.inference_mode():
    for row, p in zip(manifest['selected'], paths):
        image = read_img_seq([str(p)])[0]
        assert tuple(image.shape) == (3, 400, 600)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        y = forward(image)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        assert y.shape == (400, 600, 3) and np.isfinite(y).all()
        assert y.min() >= 0 and y.max() <= 1
        np.save(out / (p.stem + '.npy'), y)
        rows.append({'low': row['low'], 'shape': list(y.shape), 'min': float(y.min()), 'max': float(y.max()),
                     'output_sha256': hashlib.sha256(y.tobytes()).hexdigest(),
                     'runtime_s': elapsed, 'peak_memory_bytes': torch.cuda.max_memory_allocated()})
        print(row['low'], elapsed, flush=True)
after = parameter_hash(model)
assert before == after
assert decoded == list(map(str, paths))
(out / 'receipt.json').write_text(json.dumps({'mode': 'ttie_native_pad16',
    'reference': 'official VideoBaseModel.test + extracted official low-only5x5blur; not test4',
    'decoded_paths': decoded, 'denied_reads': denied, 'parameter_hash_before': before,
    'parameter_hash_after': after, 'pad_lrtb': [0, 8, 0, 0], 'blur': 'native OpenCV5x5defaultborder, then reflect-pad low+feature',
    'rows': rows}, indent=2) + '\n')
print('BASELINE PASS:8 independent native-pad16 forwards', flush=True)
