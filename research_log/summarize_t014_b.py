"""Render the completed, verified fresh result without changing its protocol."""
import argparse
from collections import Counter
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('audit', type=Path)
a = p.parse_args()
read = lambda name: json.loads((a.audit / name).read_text())
s = read('summary.json')
remote = read('output_verification.json')
local = read('local_verification.json')
primary = 'region2_ttt_energy_sobolev'
clauses = [
    ('Clean mean MSE', 'clean_mean', 'clean_mean', .003),
    ('Clean p95 MSE', 'clean_p95', 'clean_p95', .005),
    ('Homogeneous dark / identity', 'dark_ratio', 'homogeneous_dark', .60),
    ('Homogeneous bright / identity', 'bright_ratio', 'homogeneous_bright', .60),
    ('Heterogeneous / global Sobolev', 'global_ratio', 'spatial_value', .85),
    ('Heterogeneous / direct', 'direct_ratio', 'beyond_direct', .95),
    ('Heterogeneous / projected discrete', 'discrete_ratio', 'beyond_discrete', .95),
    ('Heterogeneous / frozen semantic16', 'fixed_step_ratio', 'beyond_fixed_step', .95),
    ('Heterogeneous / same-source value-only', 'value_only_ratio', 'beyond_value_only', .95),
    ('Heterogeneous dark-region / identity', 'dark_region_ratio', 'dark_region_safety', 1.05),
    ('Heterogeneous bright-region / identity', 'bright_region_ratio', 'bright_region_safety', 1.05),
    ('Exact quadrants / bilinear Sobolev', 'quadrant_bilinear_ratio', 'renderer_support', .90),
]
lines = ['# T014 fresh Stage B: verified twelve-clause result', '',
    f"Qualification: **{sum(s['criteria'].values())}/12 PASS**, qualified={s['qualified']}, failed={s['failed']}.", '',
    '40 fresh images, five primary conditions (200 inputs) plus one report-only offset condition (40 inputs). Heterogeneous qualification pools only left_right and quadrants (80 inputs). Models were frozen before manifest generation; no refitting or post-outcome changes.', '',
    '| Clause | Observed | Required | Result |', '|---|---:|---:|---|']
for label, key, criterion, threshold in clauses:
    lines.append(f"| {label} | {s['values'][key]:.16g} | <= {threshold:g} | {'PASS' if s['criteria'][criterion] else 'FAIL'} |")
groups = list(s['groups'])
lines += ['', '## Full MSE table', '', '| Method | ' + ' | '.join(groups) + ' |',
          '|---|' + '---:|' * len(groups)]
for method in s['groups']['clean']:
    lines.append('| ' + method + ' | ' + ' | '.join(f"{s['groups'][g][method]['mse']['mean']:.12g}" for g in groups) + ' |')
lines += ['', 'Oracle rows use clean references only after label-free results were persisted; the oracle is not an inference method.', '',
          '## Trajectory diagnostics', '',
          'The all group includes report-only stress. primary_200 and heterogeneous_80 below separate the primary protocol from stress. Step histograms are exact counts; movable-boundary fractions exclude fixed coordinates.']
derived = {}
for method, records in s['trajectory_distributions'].items():
    records = dict(records)
    for name, conditions in [('primary_200', groups[:5]), ('heterogeneous_80', ['left_right', 'quadrants'])]:
        hist = Counter()
        keys = ['count', 'no_active', 'updates', 'projected_updates', 'movable_boundary_coordinates', 'movable_coordinates']
        d = {k: sum(records[c][k] for c in conditions) for k in keys}
        for c in conditions:
            hist.update(records[c]['selected_steps'])
        d['selected_steps'] = dict(sorted(hist.items(), key=lambda kv: int(kv[0])))
        d['projected_update_fraction'] = d['projected_updates'] / d['updates']
        d['movable_boundary_fraction'] = d['movable_boundary_coordinates'] / d['movable_coordinates']
        records[name] = d
    derived[method] = records
    lines += ['', '### ' + method, '',
              '| Group | Selected-step histogram | No-active | Projected / updates | Boundary / movable |',
              '|---|---|---:|---|---|']
    for group, d in records.items():
        lines.append(f"| {group} | {d['selected_steps']} | {d['no_active']} | {d['projected_updates']}/{d['updates']} ({d['projected_update_fraction']:.12g}) | {d['movable_boundary_coordinates']}/{d['movable_coordinates']} ({d['movable_boundary_fraction']:.12g}) |")
stress = s['groups']['offset_left_right_40']
stress_ratios = {m: stress[primary]['mse']['mean'] / values['mse']['mean']
                for m, values in stress.items() if m != primary}
lines += ['', '## Reference-only oracle and report-only offset stress', '',
          'Heterogeneous primary/oracle regret: ' + str(s['oracle']['oracle_regret_ratio']), '',
          'Heterogeneous oracle/discrete: ' + str(s['oracle']['oracle_over_discrete']), '',
          'Heterogeneous oracle/fixed16: ' + str(s['oracle']['oracle_over_fixed16']), '',
          '| Offset: primary / comparator MSE | Ratio |', '|---|---:|']
for m, ratio in stress_ratios.items():
    lines.append(f'| {m} | {ratio:.16g} |')
lines += ['', 'The offset group is excluded from all qualification clauses. These diagnostics do not modify selection, geometry, or training.', '',
          '## Verification receipts', '', '```json', json.dumps(dict(remote=remote, local=local), indent=2), '```', '']
(a.audit / 'final_distributions.md').write_text('\n'.join(lines))
(a.audit / 'final_distributions.json').write_text(json.dumps(dict(
    stage='B', source_fit_and_calibration_are_reported_separately=True,
    trajectory_distributions=derived, oracle=s['oracle'], offset_report_only=stress_ratios,
    qualification=s['criteria'], method_and_gates_unchanged=True), indent=2) + '\n')
print(json.dumps(dict(qualified=s['qualified'], primary_200=derived[primary]['primary_200'],
    heterogeneous_80=derived[primary]['heterogeneous_80'], offset_ratios=stress_ratios), indent=2))
