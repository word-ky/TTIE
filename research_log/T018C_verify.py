"""Independent artifact verifier; imports no TTIE training/evaluation implementation."""
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
from datetime import datetime, timezone
import torch
from torch import nn

torch.set_num_threads(1)
root = Path('research_log/T018C_run')
read = lambda name: json.loads((root / (name + '.json')).read_text(encoding='utf-8'))
digest = lambda raw: hashlib.sha256(raw).hexdigest()
filehash = lambda path: digest(path.read_bytes())
cfg = read('config'); receipt = read('evaluation_receipt'); freeze = read('OOF_frozen')
training = {}; refs = {}; raw_inputs = {}
for phase, entries, destination in [('train', cfg['input_artifact_hashes'], training), ('evaluation', receipt['input_artifact_hashes'], refs)]:
    for name, item in entries.items():
        raw = subprocess.check_output(['git', 'show', item['commit'] + ':' + item['path']])
        assert digest(raw) == item['sha256']
        destination[name] = json.loads(raw); raw_inputs[phase + '_' + name] = raw
for path, expected in cfg['source_code_sha256'].items():
    assert filehash(Path(path)) == expected == digest(subprocess.check_output(['git', 'show', cfg['source_sha'] + ':' + path]))

classes = [.5, .4, .6]
hard = [[x, y, 0.] for x in (.4, .5, .6) for y in (.4, .5, .6)]
grid = [[x, y, t] for x in (.4, .5, .6) for y in (.4, .5, .6) for t in (0., .05, .1)]
assert training['score_selection']['candidates'] == training['score_config']['candidates'] == hard
assert refs['config']['candidates'] == grid
assert cfg['feature_schema'] == training['score_config']['schema'] == training['fold_provenance']['base_feature_schema']
assert cfg['feature_schema']['dimension'] == len(cfg['feature_schema']['names']) == 28
assert training['fold_provenance']['input_artifact_hashes']['selection'] == {k: cfg['input_artifact_hashes']['score_selection'][k] for k in ('path', 'sha256')}
assert training['targets'] == refs['target_decisions']
assert cfg['input_artifact_hashes']['targets']['sha256'] == training['target_freeze']['decisions_sha256']
assert training['score_selection_receipt']['selection_sha256'] == cfg['input_artifact_hashes']['score_selection']['sha256']
assert training['score_selection_receipt']['config_sha256'] == cfg['input_artifact_hashes']['score_config']['sha256']
assert refs['target_decisions_frozen']['decisions_sha256'] == refs['target_report_receipt']['decisions_sha256'] == receipt['input_artifact_hashes']['target_decisions']['sha256']
assert refs['target_quantities_frozen']['quantities_sha256'] == refs['target_report_receipt']['quantities_sha256'] == receipt['input_artifact_hashes']['target_quantities']['sha256']
assert (root / 'folds.json').read_bytes() == raw_inputs['train_folds']
assert freeze['folds_sha256'] == filehash(root / 'folds.json') == '8fdb03cdac63af5d6e58b557c96679bc15c1e2e921e525916eb9d21ad22cd2c1'

expected_recipe = dict(input_dim=84, hidden=[64,64], output_dim=3, activation='SiLU', loss='unweighted_cross_entropy',
    optimizer='AdamW', lr=1e-3, weight_decay=1e-4, betas=[.9,.999], optimizer_eps=1e-8, batch_size=256, epochs=100, seed=7,
    checkpoint='final_epoch', device='cpu', normalization='training rows only; population std clamped at 1e-12', normalization_epsilon=1e-12,
    class_order=classes, training_order='seed7 randperm per epoch; original training row order')
assert cfg['recipe'] == expected_recipe
scores = training['score_selection']['episodes']; table = refs['candidate_metrics']; decisions = read('decisions'); evaluated = read('evaluation')
assert len(scores) == len(table) == len(decisions) == len(evaluated) == 120
assert len({s['episode'] for s in scores}) == 120
f = torch.tensor([s['features'] for s in scores], dtype=torch.float32)
assert f.shape == (120, 9, 28)
z = {'x': torch.cat([f[:,4], f[:,1] - f[:,4], f[:,7] - f[:,4]], 1),
     'y': torch.cat([f[:,4], f[:,3] - f[:,4], f[:,5] - f[:,4]], 1)}
for s, t in zip(scores, table):
    assert s['episode'] == t['source_directory'] and s['corners_sha256'] == t['corners_sha256']
seen = []; reconstructed = {}; head_count = 0
for fold in training['folds']:
    train = fold['train']; heldout = fold['heldout']; number = fold['fold']; seen += heldout
    assert len(train) == 96 and len(heldout) == 24
    assert set(train).isdisjoint(heldout) and sorted(train + heldout) == list(range(120))
    train_ids = sorted({table[i]['image_id'] for i in train}); heldout_ids = sorted({table[i]['image_id'] for i in heldout})
    assert train_ids == fold['train_image_ids'] and heldout_ids == fold['heldout_image_ids']
    assert len(train_ids) == 32 and len(heldout_ids) == 8 and set(train_ids).isdisjoint(heldout_ids)
    folder = root / f'fold{number}'
    fr = json.loads((folder / 'fold_frozen.json').read_text(encoding='utf-8'))
    assert filehash(folder / 'fold_frozen.json') == freeze['fold_frozen_hashes'][str(number)]
    assert filehash(folder / 'decisions.json') == fr['decisions_sha256']
    assert fr['finalized_utc'] <= freeze['finalized_utc']
    axis_predictions = {}
    for axis in ('x', 'y'):
        sub = folder / axis; head_count += 1
        r = json.loads((sub / 'receipt.json').read_text(encoding='utf-8'))
        assert r == fr['heads'][axis] and r['finalized_utc'] <= fr['finalized_utc']
        for filename, expected in r['files_sha256'].items(): assert filehash(sub / filename) == expected
        assert r['train_rows'] == train and r['heldout_rows'] == heldout and r['train_image_ids'] == train_ids and r['heldout_image_ids'] == heldout_ids
        assert r['epochs'] == 100 and r['heldout_target_read'] is False
        history = json.loads((sub / 'history.json').read_text(encoding='utf-8'))
        assert [h['epoch'] for h in history] == list(range(1, 101))
        assert r['train_cross_entropy_final'] == history[-1]['train_cross_entropy']
        checkpoint = torch.load(sub / 'head.pt', map_location='cpu', weights_only=True)
        assert checkpoint['recipe'] == expected_recipe
        state = checkpoint['state_dict']
        mean = z[axis][train].double().mean(0).float()
        scale = z[axis][train].double().std(0, unbiased=False).clamp_min(1e-12).float()
        assert torch.equal(state['x_mean'], mean) and torch.equal(state['x_scale'], scale)
        assert r['normalization'] == dict(mean=mean.tolist(), scale=scale.tolist())
        model = nn.Sequential(nn.Linear(84,64), nn.SiLU(), nn.Linear(64,64), nn.SiLU(), nn.Linear(64,3)).eval()
        model.load_state_dict({k.removeprefix('net.'): v for k,v in state.items() if k.startswith('net.')})
        with torch.no_grad(): logits = model((z[axis][heldout] - mean) / scale)
        predicted = json.loads((sub / 'predictions.json').read_text(encoding='utf-8'))
        assert predicted == dict(row_indices=heldout, logits=logits.tolist(), classes=logits.argmax(1).tolist())
        axis_predictions[axis] = predicted
    fold_decisions = []
    for j, i in enumerate(heldout):
        cx = axis_predictions['x']['classes'][j]; cy = axis_predictions['y']['classes'][j]; x = classes[cx]; y = classes[cy]
        d = dict(row_index=i, fold=number, x_logits=axis_predictions['x']['logits'][j], y_logits=axis_predictions['y']['logits'][j],
            x_class=cx, y_class=cy, bx=x, by=y, score_index=hard.index([x,y,0.]), hard_index=grid.index([x,y,0.]))
        reconstructed[i] = d; fold_decisions.append(d)
    assert fold_decisions == json.loads((folder / 'decisions.json').read_text(encoding='utf-8'))
assert head_count == freeze['head_count'] == 10 and sorted(seen) == list(range(120))
assert decisions == [reconstructed[i] for i in range(120)]

for i, (d, t, actual) in enumerate(zip(decisions, table, evaluated)):
    target = refs['target_decisions'][i]; prior = refs['target_quantities'][i]; assert target['row_index'] == i
    values = dict(zip(map(tuple, grid), t['candidate_mse'])); x = d['bx']; y = d['by']
    h0 = values[.5,.5,0.]; hs = values[x,y,0.]; best = min(values[tuple(c)] for c in hard)
    assert prior['H0'] == h0 and prior['H_star'] == best and prior['H1'] == values[target['bx'],target['by'],0.]
    expected = dict(row_index=i, reference_row_index=i, bx=x, by=y, hard_index=d['hard_index'], target_bx=target['bx'], target_by=target['by'],
        H0=h0, Hselected=hs, H_star=best, x_match=x==target['bx'], y_match=y==target['by'], joint_match=(x,y)==(target['bx'],target['by']),
        movement='no_move' if x==.5 and y==.5 else 'x_only' if y==.5 else 'y_only' if x==.5 else 'both', condition=t['condition'])
    assert actual == expected
summary = read('summary')
for name,g in summary['groups'].items():
    rows = [r for r in evaluated if name=='spatial_pool' or r['condition']==name]
    assert g['count'] == len(rows)
    m = {k:statistics.mean(r[k] for r in rows) for k in ('H0','Hselected','H_star')}; assert m == g['mse']
    assert g['ratios'] == dict(selected_over_H0=m['Hselected']/m['H0'], selected_over_H_star=m['Hselected']/m['H_star'])
    assert g['target_agreement'] == {k:dict(count=sum(r[k] for r in rows), fraction=sum(r[k] for r in rows)/len(rows)) for k in ('x_match','y_match','joint_match')}
    assert g['movements'] == {k:sum(r['movement']==k for r in rows) for k in ('no_move','x_only','y_only','both')}
    assert g['gains'] == dict(beneficial=sum(r['Hselected']<r['H0'] for r in rows), equal=sum(r['Hselected']==r['H0'] for r in rows), harmful=sum(r['Hselected']>r['H0'] for r in rows))
    assert g['harmful_examples'] == [{k:r[k] for k in ('row_index','bx','by','target_bx','target_by','H0','Hselected','H_star')} for r in rows if r['Hselected']>r['H0']]
    for axis in ('x','y'):
        matrix = [[0]*3 for _ in classes]
        for r in rows: matrix[classes.index(r['target_b'+axis])][classes.index(r['b'+axis])] += 1
        assert matrix == g['confusion_matrices'][axis]
p=summary['groups']['spatial_pool']['mse']; o=summary['groups']['offset_left_right_40']['mse']; l=summary['groups']['left_right']['mse']; q=summary['groups']['quadrants']['mse']
vector=[p['Hselected']<=.97*p['H0'], p['Hselected']<=1.03*p['H_star'], o['Hselected']<=.95*o['H0'], l['Hselected']<=1.01*l['H0'], q['Hselected']<=1.01*q['H0']]
assert list(summary['clauses'].values()) == vector and summary['passed'] == sum(vector)
assert summary['interpretation'] == ('direct_direction_probe_viable' if all(vector) else 'direct_direction_probe_insufficient')
assert freeze['finalized_utc'] < receipt['reference_opened_utc'] <= receipt['completed_utc']
assert freeze['decisions_sha256'] == receipt['decisions_sha256'] == filehash(root/'decisions.json')
assert freeze['config_sha256'] == filehash(root/'config.json')
assert receipt['oof_frozen_sha256'] == filehash(root/'OOF_frozen.json')
assert freeze['heldout_reference_evaluated'] is False and summary['development_only'] is True and summary['fresh_qualified'] is False
result=dict(passed=True, completed_utc=datetime.now(timezone.utc).isoformat(), source_hashes=len(cfg['source_code_sha256']), training_input_hashes=len(training),
    postfreeze_input_hashes=len(refs), exact_saved_heads=10, exact_oof_predictions=120, exact_episode_identity_and_reference_arithmetic=120,
    original_fold_bytes=True, train_only_normalization_exact=True, image_groups_excluded=True, all_group_diagnostics_exact=True,
    oof_frozen_before_reference_open=True, decisions_unchanged=True, formal_retraining=False, clauses=vector, literal_passed=sum(vector), interpretation=summary['interpretation'])
Path('research_log/T018C_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))
