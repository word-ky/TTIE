"""One exporter call per smoke low; separate synthetic target counterfactual."""
import hashlib
import json
import platform
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from T027A_official_path import official_forward
from ttie import retinex_exporter as exporter

root = Path(sys.argv[1])
manifest = json.loads((root / 'smoke_manifest.json').read_text())
paths = [(root / 'low' / row['low']).resolve() for row in manifest['selected']]
allowed = set(paths)
decoded, denied = [], []
read = cv2.imread


def check(path):
    p = Path(path).resolve()
    if p not in allowed:
        denied.append(str(p))
        raise PermissionError('Image outside smoke allowlist: ' + str(p))
    return p


def low_read(path, *a, **kw):
    p = check(path)
    decoded.append(str(p))
    return read(str(p), *a, **kw)


def audit(event, args):
    if event == 'open' and isinstance(args[0], str):
        p = Path(args[0])
        if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif'}:
            # Canary creation is allowed; every read is denied outside the lows.
            mode = args[1]
            if not isinstance(mode, str) or not any(c in mode for c in 'wax'):
                check(p)


cv2.imread = low_read
sys.addaudithook(audit)
config = root / 'export_config.json'
network = yaml.safe_load((root / 'official/Options/RetinexFormer_LOL_v2_real.yml').read_text())['network_g'].copy()
assert network.pop('type') == 'RetinexFormer'
config.write_text(json.dumps({'architecture': str(root / 'official/basicsr/models/archs/RetinexFormer_arch.py'),
                             'network': network}, indent=2) + '\n')
out = root / 'exporter'
sys.argv = ['retinex_exporter', '--low', *map(str, paths), '--checkpoint',
            str(root / 'LOL_v2_real.pth'), '--config', str(config), '--out', str(out)]
exporter.main()
base = json.loads((root / 'baseline/receipt.json').read_text())
receipt = json.loads((out / 'receipt.json').read_text())
assert decoded == list(map(str, paths)) == base['decoded_paths']
rows = []
for old, new, path in zip(base['rows'], receipt['rows'], paths):
    a = np.load(root / 'baseline' / (path.stem + '.npy'))
    b = np.load(out / (path.stem + '.npy'))
    assert a.shape == b.shape == (400, 600, 3)
    assert a.dtype == b.dtype == np.float32 and np.isfinite(a).all() and np.isfinite(b).all()
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    assert diff.max() <= 1e-6
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['selected'][len(rows)]['sha256']
    rows.append({'low': old['low'], 'max_abs_float_difference': float(diff.max()),
                 'mean_abs_float_difference': float(diff.mean()),
                 'official_output_sha256': old['output_sha256'], 'exporter_output_sha256': new['output_sha256']})

# Real cohort forwards are never repeated. This separate generated tensor tests
# both actual pretrained forward paths under changed/withheld inaccessible files.
model = exporter.load_model(root / 'LOL_v2_real.pth', config)
official, _ = official_forward(root / 'official')
x = torch.linspace(0, 1, 3 * 32 * 40, device='cuda').reshape(1, 3, 32, 40)
canary = root / 'counterfactual_normal.png'
counterfactual = []
with torch.inference_mode():
    for condition, content in [('absent', None), ('mutation_a', b'not-a-real-target-A'),
                               ('mutation_b', b'not-a-real-target-B'), ('withheld', None)]:
        if content is None:
            canary.unlink(missing_ok=True)
        else:
            canary.write_bytes(content)
        for opener in (cv2.imread, lambda p: open(p, 'rb')):
            try:
                opener(str(canary))
            except PermissionError:
                pass
            else:
                raise AssertionError('Canary read was not denied')
        a, b = official(x, model), exporter.forward(x, model)
        counterfactual.append({'condition': condition,
            'official_sha256': hashlib.sha256(a.tobytes()).hexdigest(),
            'exporter_sha256': hashlib.sha256(b.tobytes()).hexdigest(),
            'max_abs_difference': float(np.max(np.abs(a-b)))})
assert len({r['official_sha256'] for r in counterfactual}) == 1
assert len({r['exporter_sha256'] for r in counterfactual}) == 1
assert all(r['max_abs_difference'] <= 1e-6 for r in counterfactual)

def runtime(rows):
    v = np.array([r['runtime_s'] for r in rows])
    return {'mean_s': float(v.mean()), 'median_s': float(np.median(v)),
            'p95_s': float(np.quantile(v, .95)),
            'peak_memory_bytes': max(r['peak_memory_bytes'] for r in rows)}

summary = {'verdict': 'exporter-ready', 'rows': rows, 'decoded_paths': decoded,
           'denied_canary_reads': denied, 'counterfactual': counterfactual,
           'counterfactual_scope': 'generated32x40tensor, pretrained network; no real smoke low repeated',
           'official_runtime': runtime(base['rows']), 'exporter_runtime': runtime(receipt['rows']),
           'timing_scope': 'CUDA input ready through float CPU output, includes first-call initialization; excludes image decoding/model load',
           'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                           'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
                           'numpy': np.__version__, 'opencv': cv2.__version__},
           'checkpoint_sha256': hashlib.sha256((root / 'LOL_v2_real.pth').read_bytes()).hexdigest(),
           'GT_mean': False, 'self_ensemble': False,
           'validation_decodes': 0, 'test_decodes': 0, 'normal_decodes': 0}
(root / 'T027A_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2), flush=True)
