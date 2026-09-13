import inspect
import json
from pathlib import Path
from unittest.mock import patch
import torch
from ttie.direction_probe import CLASSES, DirectionHead
from ttie.direction_selector import CROSS_NAMES, DirectionSelector, cross_features, load_selector
from ttie.deadband_selector_run import train, SOURCES
from ttie.local_geometry import sha


def test_exact_two_fits_and_frozen_api_ignores_reference_metadata(tmp_path):
    torch.manual_seed(8)
    cross={name:torch.randn(120,28).tolist() for name in CROSS_NAMES}
    targets={i:dict(bx=CLASSES[i%3],by=CLASSES[(i+1)%3]) for i in range(120)}
    inputs=(cross,targets,dict(dimension=28,names=list(map(str,range(28)))),{},'donor-source')
    calls=[]
    def fit(z,labels):
        calls.append((z.clone(),labels.tolist()));head=DirectionHead()
        with torch.no_grad():
            head.x_mean.copy_(z.double().mean(0));head.x_scale.copy_(z.double().std(0,unbiased=False).clamp_min(1e-12))
        return head,[dict(epoch=i,train_cross_entropy=1.) for i in range(1,101)]
    output=tmp_path/'run';code={p:sha(Path(p)) for p in SOURCES}
    with patch('ttie.deadband_selector_run.verify_source',return_value=code),patch('ttie.deadband_selector_run.training_inputs',return_value=inputs),patch('ttie.deadband_selector_run.fit_head',side_effect=fit):
        train(output,'fixture')
    assert len(calls)==2
    for a,axis in enumerate(('x','y')):
        assert torch.equal(calls[a][0],cross_features(**cross)[a])
        assert calls[a][1]==[CLASSES.index(targets[i]['b'+axis]) for i in range(120)]
    receipt=json.loads((output/'selector_frozen.json').read_text())
    assert receipt['fit_calls']==dict(x=1,y=1) and receipt['head_count']==2
    assert receipt['runtime']['device']=='cpu' and receipt['engineering_freeze_only']
    for name,value in receipt['files_sha256'].items():assert sha(output/name)==value
    assert list(inspect.signature(DirectionSelector.predict).parameters)==['self',*CROSS_NAMES]
    pinned=sha(output/'selector_frozen.json')
    expected=json.loads((output/'replay_expected.json').read_text())
    forbidden=['targets.json','reference_mse.json','condition.json','family.json','image_ids.json','degradation_mask.json','gains.json','oracle.json']
    real_read=Path.read_bytes
    def guarded(path):
        assert path.name not in forbidden
        return real_read(path)
    with patch.object(Path,'read_bytes',guarded),patch('subprocess.check_output',side_effect=AssertionError('No Git/data access in inference')):
        for value in ('{"value":1}','{"value":999}'):
            for name in forbidden:(output/name).write_text(value)
            assert load_selector(output,pinned).predict(**cross)==expected
        for name in forbidden:(output/name).unlink()
        assert load_selector(output,pinned).predict(**cross)==expected
    assert sha(output/'selector_frozen.json')==pinned
