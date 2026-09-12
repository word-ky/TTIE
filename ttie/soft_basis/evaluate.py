"""Reference-only evaluation of an already finalized selection. No rendering or rescoring."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import math
import statistics
import torch
from ..routing.provenance import verify_source
from ..stop_receipt import sha
from .score import SOURCES, read, write
from .selection import HARD_INDICES, BOUNDARIES

CONDITIONS = ('left_right', 'quadrants', 'offset_left_right_40')


def ratio(x, y): return x/y if y else None


def distribution(values):
    kept = [x for x in values if x is not None]
    return dict(count=len(values), null_count=len(values)-len(kept),
        mean=statistics.mean(kept) if kept else None,
        quantiles=dict(zip(('min', 'p05', 'p25', 'p50', 'p75', 'p95', 'max'),
            torch.quantile(torch.tensor(kept, dtype=torch.float64),
                torch.tensor([0., .05, .25, .5, .75, .95, 1.], dtype=torch.float64)).tolist())) if kept else {})


def ranks(values):
    return [sum(y < x for y in values)+(sum(y == x for y in values)+1)/2 for x in values]


def spearman(a, b):
    x, y = ranks(a), ranks(b); mx, my = statistics.mean(x), statistics.mean(y)
    numerator = sum((u-mx)*(v-my) for u, v in zip(x, y))
    denominator = math.sqrt(sum((u-mx)**2 for u in x)*sum((v-my)**2 for v in y))
    return ratio(numerator, denominator)


def evaluate(selection, reference):
    refs = {r['source_directory']: r for r in reference}
    assert {r['episode'] for r in selection['episodes']} == set(refs)
    rows = []
    for selected in selection['episodes']:
        r = refs[selected['episode']]
        assert selected['corners_sha256'] == r['corners_sha256']
        mse = [r['candidate_mse'][i] for i in HARD_INDICES]
        oracle = min(range(9), key=lambda i: (mse[i], i)); chosen = selected['selected_index']
        rows.append(dict(episode=selected['episode'], image_id=r['image_id'], condition=r['condition'],
            selected_index=chosen, hard_oracle_index=oracle, selected_mse=mse[chosen],
            hard_oracle_mse=mse[oracle], full_oracle_mse=min(r['candidate_mse']),
            region2_mse=r['region2_mse'], t015_oracle_mse=r['t015_oracle_mse'],
            reference_mse=mse, energies=selected['energies'], margin=selected['margin'],
            energy_minimum_ties=selected['minimum_ties'], oracle_minimum_ties=mse.count(min(mse)),
            disagreement=chosen != oracle, outside_oracle_tie_set=mse[chosen] != min(mse),
            regret=mse[chosen]-min(mse), spearman=spearman(selected['energies'], mse),
            energy_unique_values=len(set(selected['energies'])), mse_unique_values=len(set(mse))))
    groups = {}
    for condition in ('spatial_pool', *CONDITIONS):
        cases = rows if condition == 'spatial_pool' else [r for r in rows if r['condition'] == condition]
        g = {k: statistics.mean(r[k] for r in cases) for k in
             ('selected_mse', 'region2_mse', 'hard_oracle_mse', 'full_oracle_mse', 't015_oracle_mse')}
        g.update(count=len(cases), ratios={f'selected_over_{k}': ratio(g['selected_mse'], g[k+'_mse'])
            for k in ('region2', 'hard_oracle', 'full_oracle', 't015_oracle')},
            hard_oracle_over_full_oracle=ratio(g['hard_oracle_mse'], g['full_oracle_mse']),
            selected_counts=[sum(r['selected_index'] == i for r in cases) for i in range(9)],
            oracle_counts=[sum(r['hard_oracle_index'] == i for r in cases) for i in range(9)],
            disagreement_rate=statistics.mean(r['disagreement'] for r in cases),
            outside_oracle_tie_set_rate=statistics.mean(r['outside_oracle_tie_set'] for r in cases),
            energy_minimum_tied_episodes=sum(r['energy_minimum_ties'] > 1 for r in cases),
            oracle_minimum_tied_episodes=sum(r['oracle_minimum_ties'] > 1 for r in cases),
            energy_any_tied_episodes=sum(r['energy_unique_values'] < 9 for r in cases),
            mse_any_tied_episodes=sum(r['mse_unique_values'] < 9 for r in cases),
            energy_margin=distribution([r['margin'] for r in cases]),
            spearman=distribution([r['spearman'] for r in cases]),
            regret_by_selected_boundary=[dict(index=i, **distribution([r['regret'] for r in cases if r['selected_index'] == i])) for i in range(9)])
        groups[condition] = g
    all_cases = groups['spatial_pool']; lr = groups['left_right']; q = groups['quadrants']; offset = groups['offset_left_right_40']
    clauses = dict(spatial_improves_region2_3pct=all_cases['selected_mse'] <= .97*all_cases['region2_mse'],
        spatial_within_hard_oracle_5pct=all_cases['selected_mse'] <= 1.05*all_cases['hard_oracle_mse'],
        offset_improves_region2_5pct=offset['selected_mse'] <= .95*offset['region2_mse'],
        left_right_no_more_than_1pct_worse=lr['selected_mse'] <= 1.01*lr['region2_mse'],
        quadrants_no_more_than_1pct_worse=q['selected_mse'] <= 1.01*q['region2_mse'])
    return dict(groups=groups, clauses=clauses, passed=sum(clauses.values()),
                verdict='label_free_boundary_selection_evidence' if all(clauses.values()) else 'negative',
                candidates=BOUNDARIES), rows


def markdown(report, receipt):
    lines = ['# T016-B: frozen Sobolev hard-boundary selection audit', '',
             f"Verdict: **{report['verdict']}**, {report['passed']}/5 clauses pass.", '',
             'Development-only reuse of 120 old episodes; nine hard boundaries and fixed saved corner actions. No training, optimization or new images.', '',
             f"Selection finalized and hashed before reference access: `{receipt}`", '',
             '| Group | Selected MSE | Region2 | Hard oracle | Full oracle | T015 oracle | Selected/Region2 | Selected/hard oracle | Selected/full oracle | Selected/T015 oracle | Hard/full oracle |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    fmt = lambda x: 'null' if x is None else f'{x:.12g}'
    for name, g in report['groups'].items():
        values = [g[k] for k in ('selected_mse', 'region2_mse', 'hard_oracle_mse', 'full_oracle_mse', 't015_oracle_mse')]
        lines.append('| '+name+' | '+' | '.join(map(fmt, [*values, *g['ratios'].values(), g['hard_oracle_over_full_oracle']]))+' |')
    lines += ['', '## Fixed acceptance clauses', '']
    lines += [f'- {key}: {value}' for key, value in report['clauses'].items()]
    lines += ['', '## Counts, ranks, margins and regret', '',
              'Candidate order: '+str(BOUNDARIES)+'. Exact energy and oracle ties choose first lexicographic index. Spearman uses average ranks; constant ranks give null. Null denominator/counts stay explicit. Regret is selected MSE minus nine-hard oracle MSE.', '']
    for name, g in report['groups'].items():
        lines += [f'### {name}', '', f"Selected counts: {g['selected_counts']}; oracle counts: {g['oracle_counts']}.", '',
            f"Disagreement: {g['disagreement_rate']}; outside oracle tie set: {g['outside_oracle_tie_set_rate']}. Energy/oracle minimum-tied episodes: {g['energy_minimum_tied_episodes']}/{g['oracle_minimum_tied_episodes']}.", '',
            f"Energy margin: {g['energy_margin']}", '', f"Spearman: {g['spearman']}", '',
            '| Selected index | Count | Mean regret | Regret quantiles |', '|---|---:|---:|---|']
        lines += [f"| {r['index']} | {r['count']} | {fmt(r['mean'])} | {r['quantiles']} |" for r in g['regret_by_selected_boundary']]
        lines.append('')
    lines += ['No reference information enters the scorer. Evaluation reads only immutable saved selections and accepted T016-A MSE values; no rendering occurs in evaluation.', '',
              'Stop for research-lead review. No recalibration, feature extension, boundary predictor, new candidate family or fresh evaluation was started.', '']
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    for key in ('selection-dir', 'reference-table', 'output'): p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--source-sha', required=True); a = p.parse_args()
    code = verify_source(a.source_sha, SOURCES)
    receipt = read(a.selection_dir/'selection_receipt.json')
    assert sha(a.selection_dir/'selection.json') == receipt['selection_sha256']
    assert sha(a.selection_dir/'config.json') == receipt['config_sha256']
    selection = read(a.selection_dir/'selection.json')
    reference_opened = datetime.now(timezone.utc).isoformat()
    report, rows = evaluate(selection, read(a.reference_table))
    a.output.mkdir(parents=True, exist_ok=True)
    write(a.output/'evaluation.json', rows); write(a.output/'summary.json', report)
    write(a.output/'evaluation_receipt.json', dict(source_sha=a.source_sha, source_code_sha256=code,
          reference_table_sha256=sha(a.reference_table), selection_receipt=receipt,
          reference_opened_utc=reference_opened, selection_unchanged=sha(a.selection_dir/'selection.json') == receipt['selection_sha256']))
    (a.output/'T016B_analysis.md').write_text(markdown(report, receipt), encoding='utf-8')
    print('T016-B', report['verdict'], report['passed'], '/5', flush=True)


if __name__ == '__main__': main()
