"""Offline T016-A: render saved actions, check nesting, then report MSE oracles."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import torch
from ..natural import load_image, degrade
from ..routing.provenance import T015_SOURCES, verify_source
from .renderer import CANDIDATES, HARD, FixedCorners

CONDITIONS = ('left_right', 'quadrants', 'offset_left_right_40')
METHOD = 'region2_ttt_energy_sobolev'
SOURCES = T015_SOURCES + ('ttie/soft_basis/__init__.py',
                         'ttie/soft_basis/renderer.py', 'ttie/soft_basis/screen.py')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def grid_sha(grid):
    return hashlib.sha256(grid.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def ratio(x, y):
    return x/y if y else None


def distribution(values):
    kept = [v for v in values if v is not None]
    return dict(count=len(values), zero_denominators=len(values)-len(kept),
                mean=statistics.mean(kept) if kept else None,
                quantiles=dict(zip(('min', 'p05', 'p25', 'p50', 'p75', 'p95', 'max'),
                    torch.quantile(torch.tensor(kept, dtype=torch.float64),
                        torch.tensor([0., .05, .25, .5, .75, .95, 1.], dtype=torch.float64)).tolist())) if kept else {})


def summarize(rows):
    fixed = [statistics.mean(r['candidate_mse'][i] for r in rows) for i in range(27)]
    best = min(range(27), key=lambda i: (fixed[i], i))
    groups = {}
    for condition in ('spatial_pool', *CONDITIONS):
        selected = rows if condition == 'spatial_pool' else [r for r in rows if r['condition'] == condition]
        reg = statistics.mean(r['region2_mse'] for r in selected)
        old = statistics.mean(r['t015_oracle_mse'] for r in selected)
        means = [statistics.mean(r['candidate_mse'][i] for r in selected) for i in range(27)]
        oracle = statistics.mean(min(r['candidate_mse']) for r in selected)
        gains = [r['region2_mse']-min(r['candidate_mse']) for r in selected]
        counts = [sum(min(range(27), key=lambda j: (r['candidate_mse'][j], j)) == i
                      for r in selected) for i in range(27)]
        groups[condition] = dict(count=len(selected), region2_mse=reg,
            t015_oracle_mse=old, fixed_candidate_mse=means, best_fixed_soft_mse=means[best],
            oracle_soft_mse=oracle, oracle_counts=counts,
            ratios=dict(best_fixed_over_region2=ratio(means[best], reg),
                oracle_over_region2=ratio(oracle, reg),
                oracle_over_best_fixed=ratio(oracle, means[best]),
                oracle_over_t015_oracle=ratio(oracle, old)),
            absolute_gain=distribution(gains),
            relative_gain=distribution([ratio(g, r['region2_mse']) for g, r in zip(gains, selected)]))
    g = groups['spatial_pool']
    clauses = dict(oracle_improves_region2_5pct=g['oracle_soft_mse'] <= .95*g['region2_mse'],
        oracle_improves_old_oracle_3pct=g['oracle_soft_mse'] <= .97*g['t015_oracle_mse'],
        oracle_improves_best_fixed_3pct=g['oracle_soft_mse'] <= .97*g['best_fixed_soft_mse'],
        fixed_improves_region2_5pct=g['best_fixed_soft_mse'] <= .95*g['region2_mse'])
    strong = all(clauses[k] for k in ('oracle_improves_region2_5pct',
                                    'oracle_improves_old_oracle_3pct', 'oracle_improves_best_fixed_3pct'))
    fixed_evidence = clauses['fixed_improves_region2_5pct'] and not clauses['oracle_improves_best_fixed_3pct']
    verdict = ('strong_adaptive_basis_evidence' if strong else
               'fixed_continuous_basis_evidence' if fixed_evidence else 'negative_or_inconclusive')
    return dict(best_fixed_index=best, best_fixed_candidate=CANDIDATES[best],
                per_condition_uses_global_best_fixed=True, groups=groups, clauses=clauses, verdict=verdict)


def markdown(report, sanity, code_sha):
    lines = ['# T016-A: fixed-action renderer-transfer screen', '',
        f"Result: **{report['verdict']}**. Development-only, 120 existing episodes and exactly 27 renderers.", '',
        f'Code commit: `{code_sha}`. No CLIP, energy, checkpoint selection, optimizer, projection or action fitting was rerun.', '',
        'This measures transfer of saved Region2 corner actions, not the capacity of an independently optimized soft basis.', '',
        '## Sanity checks', '', f'`{sanity}`', '',
        'Every renderer receives the identical saved physical corner grid. Exact identity is preserved. All nesting checks passed before candidate headroom was computed.', '',
        f"Global best fixed index: {report['best_fixed_index']}; (bx, by, tau) = {report['best_fixed_candidate']}. This same global choice is used in all per-condition ratios.", '',
        '| Group | Region2 | Best fixed soft | Oracle soft | T015 oracle | Fixed/Region2 | Oracle/Region2 | Oracle/fixed | Oracle/T015 oracle |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    fmt = lambda x: 'null' if x is None else f'{x:.12g}'
    for name, g in report['groups'].items():
        vals = [g[k] for k in ('region2_mse', 'best_fixed_soft_mse', 'oracle_soft_mse', 't015_oracle_mse')]
        lines.append('| '+name+' | '+' | '.join(map(fmt, [*vals, *g['ratios'].values()]))+' |')
    lines += ['', '## All fixed renderers and oracle selection counts', '',
        'Candidate and exact-MSE tie order is bx ascending, then by ascending, then tau ascending. All candidates, including hard/shifted candidates, remain in the predeclared family.', '',
        '| Index | bx/by/tau | Spatial MSE | LR MSE | Quadrants MSE | Offset MSE | Oracle counts (all/LR/Q/offset) |',
        '|---|---|---:|---:|---:|---:|---|']
    for i, candidate in enumerate(CANDIDATES):
        gs = list(report['groups'].values())
        lines.append(f'| {i} | {candidate} | '+' | '.join(fmt(g['fixed_candidate_mse'][i]) for g in gs)+
                     ' | '+' / '.join(str(g['oracle_counts'][i]) for g in gs)+' |')
    lines += ['', '## Gain over accepted hard Region2', '',
        'Gain is Region2 MSE minus oracle-soft MSE; relative gain divides by Region2 MSE. Zero denominators stay null and are counted.', '',
        '| Group | Absolute gain mean | Relative gain mean | Relative gain quantiles | Zero denominators |',
        '|---|---:|---:|---|---:|']
    for name, g in report['groups'].items():
        lines.append(f"| {name} | {fmt(g['absolute_gain']['mean'])} | {fmt(g['relative_gain']['mean'])} | {g['relative_gain']['quantiles']} | {g['relative_gain']['zero_denominators']} |")
    lines += ['', '## Literal interpretation', '', f"Threshold clauses: `{report['clauses']}`.", '',
        'Strong adaptive-basis evidence requires all three oracle clauses. Fixed-continuous-basis evidence requires >=5% fixed-renderer improvement but <3% oracle advantage over that fixed renderer. Otherwise this screen is negative/inconclusive; none of these outcomes establishes that independently optimized soft actions are impossible.', '',
        'Full 120×27 MSEs, fixed corner values/hashes, source-file hashes and per-episode gains are in candidate_metrics.json. Source/config hashes and the accepted data paths are in config.json. No candidate images were saved and accepted T015 files were not modified.', '',
        'Stop for research-lead review. No reference-optimized actions, learned basis, new images or next experiment has been started.', '']
    return '\n'.join(lines)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    for key in ('audit', 'images', 'manifest', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--source-sha', required=True)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    code = verify_source(a.source_sha, SOURCES)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))
    manifest = read(a.manifest)
    entries = [e for e in read(a.audit/'artifact_manifest.json') if e['condition'] in CONDITIONS]
    assert len(entries) == 120 and len(manifest['images']) == 40
    assert {(e['image_id'], e['condition']) for e in entries} == {
        (i['image_id'], c) for i in manifest['images'] for c in CONDITIONS}
    a.output.mkdir(parents=True, exist_ok=True)
    config = dict(task='T016-A', source_sha=a.source_sha, source_code_sha256=code,
        device=a.device, reference_only_development=True, candidates=CANDIDATES,
        candidate_tie_order='bx then by then tau ascending', images=str(a.images),
        accepted_audit=str(a.audit), manifest_sha256=sha(a.manifest),
        source_audit_hashes={name: sha(a.audit/name) for name in
            ('artifact_manifest.json', 'config.json', 'metrics.json', 'summary.json')},
        optimization_steps=0, new_image_ids=0)
    write(a.output/'config.json', config)
    clean_images = {}
    for row in manifest['images']:
        path = a.images/row['filename']
        assert sha(path) == row['sha256']
        clean_images[row['image_id']] = load_image(path)
    sanity = dict(inputs=0, max_pixel_abs=0., max_mse_diff=0.,
                  identity_checks=0, saved_corner_checks=0, passed=False)
    records = []
    # Complete the nesting/identity checks on all inputs before evaluating headroom.
    for entry in entries:
        d = a.audit/entry['directory']
        files = {'outputs.pt': entry['files']['episode']['outputs.pt'],
                 METHOD+'/trajectory.pt': entry['files'][METHOD]['trajectory.pt']}
        for name, receipt in files.items():
            assert sha(d/name) == receipt['sha256']
        outputs = torch.load(d/'outputs.pt', mmap=True, weights_only=True)
        grid = outputs[METHOD]['grid'].clone()
        step = entry['selections'][METHOD]['selected_step']
        trajectory = torch.load(d/METHOD/'trajectory.pt', mmap=True, weights_only=True)
        assert torch.equal(grid, trajectory['grids'][step])
        sanity['saved_corner_checks'] += 1
        metrics = {r['method']: r['mse'] for r in read(d/'metrics.json')}
        clean = clean_images[entry['image_id']]
        image = degrade(clean.to(a.device), entry['condition'])
        nested = FixedCorners(grid.to(image), CANDIDATES[HARD])(image).cpu()
        pixel_diff = float((nested-outputs[METHOD]['image']).abs().max())
        mse_diff = abs(float((nested-clean).square().mean())-metrics[METHOD])
        sanity['max_pixel_abs'] = max(sanity['max_pixel_abs'], pixel_diff)
        sanity['max_mse_diff'] = max(sanity['max_mse_diff'], mse_diff)
        sanity['inputs'] += 1
        write(a.output/'sanity.json', sanity)
        assert pixel_diff <= 1e-6 and mse_diff <= 1e-10, (entry['directory'], pixel_diff, mse_diff)
        identity = torch.zeros_like(grid).to(image)
        identity[:, 1] = 1
        for candidate in CANDIDATES:
            assert torch.equal(FixedCorners(identity, candidate)(image), image)
            sanity['identity_checks'] += 1
        records.append(dict(image_id=entry['image_id'], condition=entry['condition'],
            source_directory=entry['directory'], selected_step=step, corners=grid.tolist(),
            corners_sha256=grid_sha(grid), source_files={k: v['sha256'] for k, v in files.items()},
            source_metrics_sha256=sha(d/'metrics.json'), region2_mse=metrics[METHOD],
            t015_oracle_mse=metrics['oracle_best_basis']))
        if len(records) % 20 == 0:
            print('T016-A sanity', len(records), flush=True)
    sanity['passed'] = True
    write(a.output/'sanity.json', sanity)
    rows = []
    for record in records:
        clean = clean_images[record['image_id']]
        image = degrade(clean.to(a.device), record['condition'])
        grid = torch.tensor(record['corners'], dtype=torch.float32, device=a.device)
        assert grid_sha(grid) == record['corners_sha256']
        candidates = []
        for candidate in CANDIDATES:
            model = FixedCorners(grid, candidate)
            candidates.append(model(image).cpu())
            assert grid_sha(model.corners) == record['corners_sha256']
        # References enter only after this episode's complete fixed candidate render.
        mses = [float((pixels-clean).square().mean()) for pixels in candidates]
        winner = min(range(27), key=lambda i: (mses[i], i))
        gain = record['region2_mse']-mses[winner]
        rows.append(dict(record, candidate_mse=mses, oracle_index=winner,
                         absolute_gain=gain, relative_gain=ratio(gain, record['region2_mse'])))
        if len(rows) % 20 == 0:
            write(a.output/'candidate_metrics.json', rows)
            print('T016-A rendered', len(rows), flush=True)
    report = summarize(rows)
    write(a.output/'summary.json', report)
    (a.output/'T016A_analysis.md').write_text(markdown(report, sanity, a.source_sha), encoding='utf-8')
    print(json.dumps(dict(verdict=report['verdict'], clauses=report['clauses'],
                         best_fixed=report['best_fixed_candidate'],
                         ratios=report['groups']['spatial_pool']['ratios'], sanity=sanity)), flush=True)


if __name__ == '__main__':
    main()
