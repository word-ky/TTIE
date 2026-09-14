"""Eight exporter forwards, independent parity and synthetic target canaries."""
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from T027B_access import install
from T027B_adapter import load_adapter, parameter_hash
from ttie import snr_exporter as exporter

root = Path(sys.argv[1])
manifest = json.loads((root / 'smoke_manifest.json').read_text())
paths = [(root / 'low' / row['low']).resolve() for row in manifest['selected']]
decoded, denied = install(paths)
network = yaml.safe_load((root / 'official/options/test/LOLv2_real.yml').read_text())['network_G'].copy()
assert network.pop('which_model_G') == 'low_light_transformer'
network['center'] = None
config = root / 'export_config.json'
config.write_text(json.dumps({'mode': 'ttie_native_pad16', 'network': network,
    'blur': {'kernel': [5, 5], 'border': 'OpenCV default REFLECT_101', 'resolution': 'native before padding'},
    'padding': {'mode': 'reflect', 'sides': 'right bottom', 'multiple': 16, 'native_lrtb': [0, 8, 0, 0]},
    'snr': {'gray_rgb': [.299, .587, .114], 'epsilon_noise': .0001,
            'epsilon_max': .0001, 'normalization': 'per-image maximum', 'clamp': [0, 1]},
    'output': 'unpad to native, clamp0..1, HWCfloat32 before quantization'}, indent=2) + '\n')
out = root / 'exporter'
sys.argv = ['snr_exporter', '--low', *map(str, paths), '--checkpoint', str(root / 'LOLv2_real.pth'),
            '--config', str(config), '--source', str(root / 'official'), '--out', str(out)]
exporter.main()
baseline = json.loads((root / 'baseline/receipt.json').read_text())
receipt = json.loads((out / 'receipt.json').read_text())
assert decoded == list(map(str, paths)) == baseline['decoded_paths']
assert baseline['parameter_hash_before'] == baseline['parameter_hash_after'] == receipt['parameter_hash_before'] == receipt['parameter_hash_after']
rows = []
for old, new, path, selection in zip(baseline['rows'], receipt['rows'], paths, manifest['selected']):
    a = np.load(root / 'baseline' / (path.stem + '.npy'))
    b = np.load(out / (path.stem + '.npy'))
    assert a.shape == b.shape == (400, 600, 3) and a.dtype == b.dtype == np.float32
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert min(a.min(), b.min()) >= 0 and max(a.max(), b.max()) <= 1
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    assert diff.max() <= 1e-6
    assert hashlib.sha256(path.read_bytes()).hexdigest() == selection['sha256'] == new['low_sha256']
    rows.append({'low': old['low'], 'max_abs_float_difference': float(diff.max()),
                 'mean_abs_float_difference': float(diff.mean()),
                 'adapter_output_sha256': old['output_sha256'], 'exporter_output_sha256': new['output_sha256']})

# Generated32x40 input only. No real smoke low is repeated for this check.
adapter_model, adapter_forward, _ = load_adapter(root)
export_model = exporter.load_model(root / 'LOLv2_real.pth', config, root / 'official')
before = [parameter_hash(adapter_model), parameter_hash(export_model)]
x = torch.linspace(0, 1, 3 * 32 * 40).reshape(3, 32, 40)
canary = root / 'synthetic_GT.png'
counterfactual = []
with torch.inference_mode():
    for condition, content in [('absent', None), ('mutation_a', b'GT-canary-A'),
                               ('mutation_b', b'GT-canary-B'), ('withheld', None)]:
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
                raise AssertionError('Target canary read not denied')
        a, b = adapter_forward(x), exporter.forward(x, export_model)
        counterfactual.append({'condition': condition,
            'adapter_sha256': hashlib.sha256(a.tobytes()).hexdigest(),
            'exporter_sha256': hashlib.sha256(b.tobytes()).hexdigest(),
            'max_abs_float_difference': float(np.max(np.abs(a-b)))})
assert len({r['adapter_sha256'] for r in counterfactual}) == 1
assert len({r['exporter_sha256'] for r in counterfactual}) == 1
assert max(r['max_abs_float_difference'] for r in counterfactual) <= 1e-6
after = [parameter_hash(adapter_model), parameter_hash(export_model)]
assert before == after

def runtime(rows):
    v = np.array([r['runtime_s'] for r in rows])
    return {'mean_s': float(v.mean()), 'median_s': float(np.median(v)), 'p95_s': float(np.quantile(v, .95)),
            'peak_memory_bytes': max(r['peak_memory_bytes'] for r in rows)}

summary = {'verdict': 'exporter-ready', 'mode': 'ttie_native_pad16',
    'interpretation': 'predeclared native padding adaptation, NOT official resize-based test4 reproduction',
    'rows': rows, 'decoded_paths': decoded, 'denied_canary_reads': denied,
    'counterfactual': counterfactual, 'counterfactual_scope': 'generated32x40tensor only; no real smoke low repeated',
    'parameter_hash_before': before, 'parameter_hash_after': after,
    'adapter_runtime': runtime(baseline['rows']), 'exporter_runtime': runtime(receipt['rows']),
    'timing_scope': 'decoded CPU RGB tensor through native blur, CUDA pad/SNR/network and CPU output; includes cold first call; excludes image decode/model load',
    'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(), 'packages': {n: importlib.metadata.version(n) for n in ['torch','numpy','opencv-python','PyYAML','pytest']}},
    'checkpoint_sha256': hashlib.sha256((root / 'LOLv2_real.pth').read_bytes()).hexdigest(),
    'normal_decodes': 0, 'validation_decodes': 0, 'test_decodes': 0}
(root / 'T027B_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2), flush=True)
