"""T019-B label-only intervention; the T018-C head lifecycle stays unchanged."""
import argparse
import hashlib
import json
import platform
import re
import subprocess
from pathlib import Path

import torch
from .direction_probe import RECIPE, axis_features
from .direction_probe_run import SOURCES as DONOR_SOURCES, train_fold, freeze_oof, confusion
from .energy_local import blob, load_scoring, load_references, join_and_measure, report_groups
from .local_geometry import now, sha, write
from .routing.provenance import verify_source

DONOR = 'b45bf9563e7c3f7e8aa27c923a35f1aabb95a5cb'
TARGET = dict(commit='91f750e871d2c133624bbe4972f3fb3086f25a6c',
    path='research_log/T019A_run/decisions.json',
    sha256='d874bed74b0ebf68b680c8b6e60c8c5a44c9d82ce3cb7fd5730e16e53a980d45')
SOURCES = DONOR_SOURCES + ('ttie/deadband_probe.py',)


def raw_blob(item):
    raw = subprocess.check_output(['git', 'show', item['commit'] + ':' + item['path']])
    assert hashlib.sha256(raw).hexdigest() == item['sha256']
    return raw


def training_targets(raw, train_indices):
    # The accepted artifact is pretty-printed with top-level objects indented two
    # spaces. Scan structural boundaries; decode only bx/by in requested rows.
    # Do not deserialize cross-MSE, nested reference gains, or held-out labels.
    entries = re.findall(rb'^  \{\r?\n.*?^  \}', raw, re.M | re.S)
    assert len(entries) == 120
    targets = {}
    for i in train_indices:
        row = entries[i]
        assert int(re.search(rb'^    "row_index": (\d+)', row, re.M)[1]) == i
        targets[i] = {key: float(re.search(rb'^    "' + key.encode() + rb'": ([0-9.]+)', row, re.M)[1])
                      for key in ('bx', 'by')}
    return targets


def inputs():
    donor, donor_origin = blob(DONOR, 'research_log/T018C_run/config.json')
    scores, hashes = load_scoring()
    hashes = {'score_' + k: v for k, v in hashes.items()}
    for key, item in hashes.items():
        assert item == donor['input_artifact_hashes'][key]
    for key in ('folds', 'fold_provenance'):
        hashes[key] = donor['input_artifact_hashes'][key]
    raw = raw_blob(hashes['folds'])
    assert RECIPE == donor['recipe']
    for path in DONOR_SOURCES:
        assert sha(Path(path)) == donor['source_code_sha256'][path]
    hashes['targets'] = TARGET
    hashes['donor_config'] = donor_origin
    return scores, json.loads(raw), raw, hashes


def train(output, source):
    code = verify_source(source, SOURCES)
    scores, folds, fold_bytes, hashes = inputs()
    target_bytes = raw_blob(TARGET)  # Whole-artifact byte hash; no label decoding.
    z = axis_features([r['features'] for r in scores['selection']['episodes']])
    output.mkdir(parents=True)
    (output / 'folds.json').write_bytes(fold_bytes)
    write(output / 'config.json', dict(task='T019-B', source_sha=source, source_code_sha256=code,
        input_artifact_hashes=hashes, feature_schema=scores['config']['schema'],
        feature_construction='concat(f0,fminus-f0,fplus-f0)', recipe=RECIPE,
        runtime=dict(python=platform.python_version(), torch=torch.__version__, device='cpu'),
        development_only=True, target_origin='accepted PR32 head; not merged into main'))
    decisions = []
    for fold in folds:
        folder = output / f"fold{fold['fold']}"
        targets = training_targets(target_bytes, fold['train'])
        decisions += train_fold(z, targets, fold, folder)
        # Add the requested explicit normalization and train-label bindings before
        # the global freeze. Head/history/prediction bytes remain donor-produced.
        fr = json.loads((folder / 'fold_frozen.json').read_text(encoding='utf-8'))
        write(folder / 'train_targets.json', targets)
        for axis in ('x', 'y'):
            receipt = fr['heads'][axis]
            write(folder / axis / 'normalization.json', receipt['normalization'])
            receipt['files_sha256']['normalization.json'] = sha(folder / axis / 'normalization.json')
            receipt['train_targets_sha256'] = sha(folder / 'train_targets.json')
            receipt['target_artifact'] = TARGET
            write(folder / axis / 'receipt.json', receipt)
        fr['finalized_utc'] = now()
        write(folder / 'fold_frozen.json', fr)
        del targets
        print('Fold', fold['fold'], 'two unchanged-recipe heads frozen', flush=True)
    freeze_oof(output, decisions)
    print('120 OOF decisions frozen', sha(output / 'decisions.json'), flush=True)


def diagnostics(rows, targets):
    result = {}
    for name in ('spatial_pool', 'left_right', 'quadrants', 'offset_left_right_40'):
        group = [r for r in rows if name == 'spatial_pool' or r['condition'] == name]
        result[name] = dict(
            predicted_classes={axis: {str(v): sum(r['b'+axis] == v for r in group) for v in (.5,.4,.6)} for axis in ('x','y')},
            predicted_joint={f'{x},{y}': sum((r['bx'],r['by']) == (x,y) for r in group) for x in (.5,.4,.6) for y in (.5,.4,.6)},
            noncenter_on_center_target={axis: sum(r['target_b'+axis] == .5 and r['b'+axis] != .5 for r in group) for axis in ('x','y')})
    suppressed = [dict(row_index=r['row_index'], axis=a, prediction=r['b'+a],
                       condition=r['condition'], original_target=targets[r['row_index']]['original_b'+a])
                  for r in rows for a in ('x','y')
                  if r['target_b'+a] == .5 and targets[r['row_index']]['original_b'+a] != .5]
    assert len(suppressed) == 26
    return dict(groups=result, suppressed_axes=dict(total=26,
        predicts_center=sum(r['prediction']==.5 for r in suppressed),
        predicts_move=sum(r['prediction']!=.5 for r in suppressed), rows=suppressed),
        quadrants_moves=[r for r in rows if r['condition']=='quadrants' and r['movement']!='no_move'])


def evaluate(output):
    read = lambda name: json.loads((output / (name + '.json')).read_text(encoding='utf-8'))
    config = read('config'); frozen = read('OOF_frozen')
    verify_source(config['source_sha'], SOURCES)
    assert sha(output/'decisions.json') == frozen['decisions_sha256']
    assert sha(output/'config.json') == frozen['config_sha256']
    assert sha(output/'folds.json') == frozen['folds_sha256']
    replay = read('head_replay')
    assert replay['passed'] and replay['decisions_sha256'] == frozen['decisions_sha256']
    opened = now()
    data, hashes = load_references()
    targets = json.loads(raw_blob(TARGET)); hashes['deadband_targets'] = TARGET
    scores, score_hashes = load_scoring()
    for k,v in score_hashes.items(): assert config['input_artifact_hashes']['score_'+k] == v
    rows = join_and_measure(read('decisions'), scores['selection']['episodes'], data)
    for row, target in zip(rows, targets):
        assert row['row_index'] == row['reference_row_index'] == target['row_index']
        for a in ('x','y'):
            row['target_b'+a] = target['b'+a]
            row[a+'_match'] = row['b'+a] == target['b'+a]
        row['joint_match'] = row['x_match'] and row['y_match']
    for fold in read('folds'):
        for part in ('train','heldout'):
            assert sorted({data['candidate_metrics'][i]['image_id'] for i in fold[part]}) == fold[part+'_image_ids']
    report = report_groups(rows)
    for name, group in report['groups'].items():
        group['confusion_matrices'] = confusion([r for r in rows if name=='spatial_pool' or r['condition']==name])
    report['clauses']['pooled_harmful_at_most_10'] = report['groups']['spatial_pool']['gains']['harmful'] <= 10
    report['clauses']['quadrants_zero_harmful'] = report['groups']['quadrants']['gains']['harmful'] == 0
    report['passed'] = sum(report['clauses'].values())
    report['interpretation'] = 'T019-B grouped-OOF deadband-direction ' + ('positive' if report['passed']==7 else 'negative')
    donor, hashes['T018C_summary'] = blob(DONOR, 'research_log/T018C_run/summary.json')
    detail = diagnostics(rows, targets)
    detail['T018C_comparison'] = {name: {key: {'T018C': donor['groups'][name][key], 'T019B': group[key]}
        for key in ('ratios','gains','movements')} for name, group in report['groups'].items()}
    write(output/'evaluation.json', rows); write(output/'summary.json', report); write(output/'diagnostics.json', detail)
    write(output/'evaluation_receipt.json', dict(reference_opened_utc=opened, completed_utc=now(),
        input_artifact_hashes=hashes, decisions_sha256=sha(output/'decisions.json'),
        oof_frozen_sha256=sha(output/'OOF_frozen.json'), head_replay_sha256=sha(output/'head_replay.json'),
        exact_episode_identity=120, actual_image_id_grouping_exact=True, T018E_data_used=False))
    print(json.dumps({k: report[k] for k in ('clauses','passed','interpretation')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['train','evaluate'])
    p.add_argument('--output', type=Path, required=True); p.add_argument('--source-sha'); a = p.parse_args()
    if a.stage == 'train': train(a.output, a.source_sha)
    else: evaluate(a.output)
