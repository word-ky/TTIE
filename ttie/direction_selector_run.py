"""T018-D: one all-development fit, immutable receipt, separate feature-only replay."""
import argparse
import json
from pathlib import Path
import platform
import torch
from .direction_probe import CLASSES, RECIPE, fit_head
from .direction_selector import CROSS_NAMES, INFERENCE_SOURCES, HARD, DirectionSelector, cross_features, load_selector
from .energy_local import blob, load_scoring, TARGET_COMMIT
from .local_geometry import now, sha, write
from .routing.provenance import verify_source

ACCEPTED_C = 'b45bf9563e7c3f7e8aa27c923a35f1aabb95a5cb'
SOURCES = tuple(dict.fromkeys(INFERENCE_SOURCES + ('ttie/direction_selector_run.py', 'ttie/energy_local.py',
    'ttie/local_geometry.py', 'ttie/hard_local.py', 'ttie/routing/__init__.py', 'ttie/routing/provenance.py')))


def training_inputs():
    scores, hashes = load_scoring(); hashes = {'score_' + k: v for k, v in hashes.items()}
    targets, hashes['targets'] = blob(TARGET_COMMIT, 'research_log/T018A_run/decisions.json')
    frozen, hashes['target_freeze'] = blob(TARGET_COMMIT, 'research_log/T018A_run/decisions_frozen.json')
    donor, hashes['accepted_c_config'] = blob(ACCEPTED_C, 'research_log/T018C_run/config.json')
    assert donor['recipe'] == RECIPE
    for name in ('score_selection', 'score_config', 'score_selection_receipt', 'targets', 'target_freeze'):
        assert donor['input_artifact_hashes'][name] == hashes[name]
    assert sha(Path('ttie/direction_probe.py')) == donor['source_code_sha256']['ttie/direction_probe.py']
    entries = scores['selection']['episodes']; schema = scores['config']['schema']
    assert len(entries) == len(targets) == 120 and len({r['episode'] for r in entries}) == 120
    assert schema == donor['feature_schema'] and schema['dimension'] == len(schema['names']) == 28
    assert hashes['targets']['sha256'] == frozen['decisions_sha256']
    assert all(t['row_index'] == i and t['hard_index'] == 3 * HARD.index((t['bx'], t['by'], 0.)) for i, t in enumerate(targets))
    assert all(len(r['features']) == 9 and all(len(v) == 28 for v in r['features']) for r in entries)
    cross = {name: [r['features'][index] for r in entries] for name, index in zip(CROSS_NAMES, (4,1,7,3,5))}
    return cross, targets, schema, hashes, [r['episode'] for r in entries]


def train(output, source):
    code = verify_source(source, SOURCES); cross, targets, schema, hashes, episodes = training_inputs()
    output.mkdir(parents=True); started = now()
    config = dict(task='T018-D', source_sha=source, source_code_sha256=code, input_artifact_hashes=hashes,
        feature_schema=schema, recipe=RECIPE, training_rows=list(range(120)), training_episode_keys=episodes,
        feature_order=list(CROSS_NAMES), cross_candidate_indices=[4,1,7,3,5],
        runtime=dict(python=platform.python_version(), torch=torch.__version__, device='cpu'), fresh_qualified=False)
    write(output/'config.json', config); write(output/'replay_features.json', cross)
    z = cross_features(**cross); heads = []; normalization = {}; history_files = []
    for axis, features in zip(('x','y'), z):
        labels = torch.tensor([CLASSES.index(t['b'+axis]) for t in targets])
        head, history = fit_head(features, labels); heads.append(head)
        torch.save(dict(state_dict=head.state_dict(), recipe=RECIPE), output/('head_'+axis+'.pt'))
        filename = 'history_'+axis+'.json'; write(output/filename, history); history_files.append(filename)
        normalization[axis] = dict(mean=head.x_mean.tolist(), scale=head.x_scale.tolist())
        print('Final', axis, 'head trained once: 120 rows, 100 epochs', flush=True)
    write(output/'normalization.json', normalization)
    write(output/'replay_expected.json', DirectionSelector(*heads).predict(**cross))
    files = ['config.json','head_x.pt','head_y.pt','normalization.json','replay_features.json','replay_expected.json'] + history_files
    receipt = dict(task='T018-D', started_utc=started, finalized_utc=now(), source_sha=source, source_code_sha256=code,
        inference_code_sha256={p:code[p] for p in INFERENCE_SOURCES}, input_artifact_hashes=hashes,
        feature_schema=schema, feature_order=list(CROSS_NAMES), feature_construction='concat(f0,fminus-f0,fplus-f0)',
        candidates=HARD, recipe=RECIPE, normalization=normalization, training_rows=120, head_count=2,
        fit_calls=dict(x=1,y=1), epochs=dict(x=100,y=100), files_sha256={f:sha(output/f) for f in files},
        replay_rows=120, replay_role='fixed accepted development features; serialization check only',
        reference_metrics_opened=False, fresh_qualified=False)
    write(output/'selector_frozen.json', receipt)
    (output/'selector_frozen.sha256').write_text(sha(output/'selector_frozen.json')+'\n', encoding='utf-8')
    print('Selector frozen', sha(output/'selector_frozen.json'), flush=True)


def replay(output, pinned):
    selector = load_selector(output, pinned)
    read = lambda name: json.loads((output/name).read_text(encoding='utf-8'))
    receipt = read('selector_frozen.json')
    for name in ('replay_features.json','replay_expected.json'):
        assert sha(output/name) == receipt['files_sha256'][name]
    decisions = selector.predict(**read('replay_features.json'))
    assert decisions == read('replay_expected.json')
    write(output/'replay_actual.json', decisions)
    write(output/'replay_receipt.json', dict(passed=True, completed_utc=now(), selector_receipt_sha256=pinned,
        replay_actual_sha256=sha(output/'replay_actual.json'), exact_rows=len(decisions), target_or_reference_reads=False))
    print('PASS: exact saved/reloaded feature-only replay of', len(decisions), 'rows')


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('stage', choices=['train','replay'])
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--source-sha'); parser.add_argument('--receipt-sha')
    args=parser.parse_args()
    if args.stage=='train': train(args.output,args.source_sha)
    else: replay(args.output,args.receipt_sha)


if __name__=='__main__': main()
