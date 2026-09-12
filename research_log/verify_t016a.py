"""Post-run saved-table/source audit; no rendering, model loading or optimization."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument('audit', type=Path)
p.add_argument('--old-audit', type=Path, required=True)
p.add_argument('--archive', type=Path, required=True)
a = p.parse_args()
read = lambda path: json.loads(path.read_text(encoding='utf-8'))
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
config = read(a.audit/'config.json')
rows = read(a.audit/'candidate_metrics.json')
report = read(a.audit/'summary.json')
sanity = read(a.audit/'sanity.json')
assert sanity == dict(inputs=120, max_pixel_abs=0., max_mse_diff=0.,
                     identity_checks=3240, saved_corner_checks=120, passed=True)
assert config['optimization_steps'] == config['new_image_ids'] == 0
assert config['source_sha'] == '05e7dcf268c7b0479e6edabdc6ec76528d5730bf'
for path, digest in config['source_code_sha256'].items():
    assert hashlib.sha256(subprocess.check_output(['git', 'show', config['source_sha']+':'+path])).hexdigest() == digest
for name, digest in config['source_audit_hashes'].items():
    assert sha(a.old_audit/name) == digest
assert sha(Path('research_log/T015_manifest.json')) == config['manifest_sha256']
conditions = ('left_right', 'quadrants', 'offset_left_right_40')
candidates = list(itertools.product((.4, .5, .6), (.4, .5, .6), (0., .05, .1)))
assert config['candidates'] == [list(c) for c in candidates]
old = [r for r in read(a.old_audit/'artifact_manifest.json') if r['condition'] in conditions]
assert len(rows) == len(old) == 120
values = np.asarray([r['candidate_mse'] for r in rows], dtype=np.float64)
assert values.shape == (120, 27) and np.isfinite(values).all()
for row, original in zip(rows, old):
    assert (row['image_id'], row['condition'], row['source_directory']) == (
        original['image_id'], original['condition'], original['directory'])
    directory = a.old_audit/original['directory']
    method = 'region2_ttt_energy_sobolev'
    step = original['selections'][method]['selected_step']
    assert step == row['selected_step']
    assert row['source_files']['outputs.pt'] == original['files']['episode']['outputs.pt']['sha256']
    path = method+'/trajectory.pt'
    assert row['source_files'][path] == sha(directory/path)
    grid = torch.load(directory/path, weights_only=True)['grids'][step]
    assert torch.equal(grid, torch.tensor(row['corners'], dtype=grid.dtype))
    assert hashlib.sha256(grid.contiguous().numpy().tobytes()).hexdigest() == row['corners_sha256']
    assert sha(directory/'metrics.json') == row['source_metrics_sha256']
    metrics = {r['method']: r['mse'] for r in read(directory/'metrics.json')}
    assert row['region2_mse'] == metrics[method] == row['candidate_mse'][12]
    assert row['t015_oracle_mse'] == metrics['oracle_best_basis']
    assert row['oracle_index'] == int(np.argmin(row['candidate_mse']))
    assert row['absolute_gain'] == row['region2_mse']-min(row['candidate_mse'])
    assert row['relative_gain'] == (row['absolute_gain']/row['region2_mse'] if row['region2_mse'] else None)
best = int(np.argmin(values.mean(0)))
assert best == report['best_fixed_index'] == 12
differences = []
def close(x, y):
    if x is None or y is None:
        assert x is y
    else:
        differences.append(abs(x-y))
        assert abs(x-y) <= 1e-12

for name in ('spatial_pool', *conditions):
    chosen = rows if name == 'spatial_pool' else [r for r in rows if r['condition'] == name]
    v = np.asarray([r['candidate_mse'] for r in chosen])
    reg = np.asarray([r['region2_mse'] for r in chosen])
    oracle = v.min(1)
    old_oracle = np.asarray([r['t015_oracle_mse'] for r in chosen])
    g = report['groups'][name]
    assert g['count'] == len(chosen)
    for x, y in zip(v.mean(0), g['fixed_candidate_mse']): close(x, y)
    for x, k in ((reg.mean(), 'region2_mse'), (old_oracle.mean(), 't015_oracle_mse'),
                 (v[:, best].mean(), 'best_fixed_soft_mse'), (oracle.mean(), 'oracle_soft_mse')):
        close(x, g[k])
    ratios = [v[:, best].mean()/reg.mean(), oracle.mean()/reg.mean(),
              oracle.mean()/v[:, best].mean(), oracle.mean()/old_oracle.mean()]
    for x, y in zip(ratios, g['ratios'].values()): close(x, y)
    assert np.bincount(v.argmin(1), minlength=27).tolist() == g['oracle_counts']
    for key, vals in (('absolute_gain', reg-oracle), ('relative_gain', (reg-oracle)/reg)):
        assert g[key]['zero_denominators'] == 0
        close(vals.mean(), g[key]['mean'])
        for x, y in zip(np.quantile(vals, [0., .05, .25, .5, .75, .95, 1.]), g[key]['quantiles'].values()): close(x, y)
g = report['groups']['spatial_pool']
assert report['clauses'] == dict(oracle_improves_region2_5pct=g['oracle_soft_mse'] <= .95*g['region2_mse'],
    oracle_improves_old_oracle_3pct=g['oracle_soft_mse'] <= .97*g['t015_oracle_mse'],
    oracle_improves_best_fixed_3pct=g['oracle_soft_mse'] <= .97*g['best_fixed_soft_mse'],
    fixed_improves_region2_5pct=g['best_fixed_soft_mse'] <= .95*g['region2_mse'])
assert all(list(report['clauses'].values())[:3]) and report['verdict'] == 'strong_adaptive_basis_evidence'
result = dict(task='T016-A', source_code_blobs_verified=len(config['source_code_sha256']),
    saved_grids_and_metric_sources_verified=120, candidate_mses=3240,
    archive_sha256=sha(a.archive), max_summary_abs_difference=max(differences),
    canonical_nesting_max_pixel_abs=0., canonical_nesting_max_mse_diff=0.,
    all_identity_checks=3240, no_optimization_or_new_images=True,
    oracle_tau_counts={str(t):sum(g['oracle_counts'][i] for i,c in enumerate(candidates) if c[2]==t) for t in (0., .05, .1)},
    episodes_with_positive_gain=sum(r['absolute_gain'] > 0 for r in rows),
    episodes_with_tied_minimum=sum(r['candidate_mse'].count(min(r['candidate_mse'])) > 1 for r in rows),
    verdict=report['verdict'])
(a.audit/'local_verification.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
print(json.dumps(result, indent=2))
