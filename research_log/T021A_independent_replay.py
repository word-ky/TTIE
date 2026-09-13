"""Independent frozen-artifact verifier; imports no TTIE/adaptation modules."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import convolve1d


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def ssim(x, y):
    # Explicit separable 11-tap kernel, independent of gaussian_filter/TTIE.
    x = x.squeeze(0).permute(1, 2, 0).numpy().astype(np.float64)
    y = y.squeeze(0).permute(1, 2, 0).numpy().astype(np.float64)
    kernel = np.exp(-np.arange(-5, 6, dtype=np.float64)**2 / (2*1.5**2))
    kernel /= kernel.sum()
    def smooth(v):
        return convolve1d(convolve1d(v, kernel, axis=0, mode='reflect'), kernel, axis=1, mode='reflect')
    mx, my = smooth(x), smooth(y)
    numerator = (2*mx*my + 0.0001) * (2*(smooth(x*y)-mx*my) + 0.0009)
    denominator = (mx*mx+my*my+0.0001) * (smooth(x*x)-mx*mx+smooth(y*y)-my*my+0.0009)
    return float(np.mean(numerator/denominator))


def main(root):
    binding = json.loads((root/'binding.json').read_bytes())
    for name, digest in json.loads((root/'artifact_hashes.json').read_bytes()).items():
        assert sha(root/name) == digest
    table = list(csv.DictReader((root/'ssim_rows.csv').open()))
    assert len(table) == len(binding['rows']) == 200
    clean = {r['image_id']: r for r in binding['rows'] if r['condition']=='clean'}
    for row in binding['rows']:
        for f in row['files'].values():
            assert sha(f['path']) == f['sha256']
    references = {i: torch.load(r['files']['outputs.pt']['path'], map_location='cpu', weights_only=True)['identity']['image'] for i,r in clean.items()}
    errors = []
    deltas = []
    for row, recorded in zip(binding['rows'], table):
        assert (str(row['image_id']), row['condition']) == (recorded['image_id'], recorded['condition'])
        outputs = torch.load(row['files']['outputs.pt']['path'], map_location='cpu', weights_only=True)
        scores = [ssim(outputs[binding[h]]['image'], references[row['image_id']]) for h in ['H0','H1']]
        errors.extend(abs(scores[i]-float(recorded[f'SSIM{i}'])) for i in range(2))
        deltas.append(scores[1]-scores[0])
    assert max(errors) < 1e-12
    d = np.array(deltas); ids = np.array([r['image_id'] for r in binding['rows']]); clusters = np.unique(ids)
    means_by_image = np.array([d[ids==i].mean() for i in clusters])
    draws = np.random.Generator(np.random.PCG64(7)).integers(0,40,(10000,40))
    assert np.array_equal(draws, np.load(root/'bootstrap_draws.npy'))
    distribution = means_by_image[draws].mean(axis=1)
    bootstrap_error = float(np.max(np.abs(distribution-np.load(root/'bootstrap_means.npy'))))
    assert bootstrap_error < 1e-12
    ci = np.quantile(distribution,[.025,.975],method='linear')
    summary = json.loads((root/'summary.json').read_bytes())
    assert np.allclose(ci, summary['ci95'], rtol=0, atol=1e-12)
    assert abs(float(d.mean())-summary['mean']) < 1e-12
    assert abs(float(np.median(d))-summary['median']) < 1e-12
    assert bool(ci[0]>0) == summary['positive']
    for condition, stat in summary['conditions'].items():
        assert abs(d[np.array([r['condition']==condition for r in table])].mean()-stat['mean']) < 1e-12
    result = dict(passed=True, rows=200, source_images=40, metric_max_abs_error=max(errors),
                  bootstrap_max_abs_error=bootstrap_error, ci95=ci.tolist(),
                  implementation='Explicit separable Gaussian kernel with scipy.convolve1d; no TTIE imports',
                  verifier_source_sha256=sha(__file__), verdict=summary['verdict'])
    (root/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
