import copy
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from ttie.deadband_probe import training_targets, evaluate
from ttie.direction_probe import axis_features, fit_axis_fold, predict
from ttie.direction_probe_run import freeze_oof
from ttie.local_geometry import write, sha


def test_train_only_extraction_and_predictions_ignore_heldout_payload():
    targets = [dict(row_index=i, bx=(.5,.4,.6)[i%3], by=.5,
                    cross_values=['MSE forbidden'], family='metadata forbidden') for i in range(120)]
    fold = dict(train=list(range(96)), heldout=list(range(96,120)))
    raw = json.dumps(targets, indent=2).encode()
    selected = training_targets(raw, fold['train'])
    assert set(selected) == set(fold['train'])
    assert all(set(t)=={'bx','by'} for t in selected.values())
    changed = copy.deepcopy(targets)
    for i in fold['heldout']:
        changed[i]['bx'] = {'not even a class': ['heldout label must not be decoded']}
        changed[i]['by'] = None
        changed[i]['cross_values'] = [1e9]; changed[i]['family'] = 'different'
    assert training_targets(json.dumps(changed, indent=2).encode(), fold['train']) == selected
    with pytest.raises(KeyError): selected[96]
    torch.manual_seed(21); z=axis_features(torch.randn(120,9,28))
    a, _, logits, classes = fit_axis_fold(z, selected, fold, 0)
    b, _, other_logits, other_classes = fit_axis_fold(z, training_targets(json.dumps(changed,indent=2).encode(),fold['train']),fold,0)
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    assert (logits,classes)==(other_logits,other_classes)==predict(a,z[fold['heldout'],0])


def test_reference_open_requires_complete_freeze_and_replay():
    with tempfile.TemporaryDirectory() as directory:
        p=Path(directory);write(p/'config.json',dict(source_sha='mock'));write(p/'folds.json',[])
        for i in range(5):
            (p/f'fold{i}').mkdir();write(p/f'fold{i}'/'fold_frozen.json',dict(fold=i))
        freeze_oof(p,[dict(row_index=i) for i in range(120)])
        class ReferenceOpened(Exception): pass
        def references():
            assert (p/'head_replay.json').exists()
            assert json.loads((p/'OOF_frozen.json').read_text())['decisions_sha256']==sha(p/'decisions.json')
            raise ReferenceOpened()
        with patch('ttie.deadband_probe.verify_source',return_value={}), patch('ttie.deadband_probe.load_references',side_effect=references) as ref:
            with pytest.raises(FileNotFoundError): evaluate(p)
            ref.assert_not_called()
            write(p/'head_replay.json',dict(passed=True,decisions_sha256=sha(p/'decisions.json')))
            with pytest.raises(ReferenceOpened): evaluate(p)
            (p/'decisions.json').write_text('[]')
            ref.reset_mock()
            with pytest.raises(AssertionError): evaluate(p)
            ref.assert_not_called()
