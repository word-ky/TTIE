import sys,inspect
from pathlib import Path
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research_log/T031A_support'))
from core import distances
from analyze import classify
def test_exact_standardized_distance_and_no_image_api(monkeypatch):
    assert list(inspect.signature(distances).parameters)==['source_features','state_features','x_mean','x_scale']
    from PIL import Image
    def forbidden(*args,**kwargs):raise AssertionError('image access')
    monkeypatch.setattr(Image,'open',forbidden)
    source=torch.stack([torch.zeros(28),torch.full((28,),4.)]);q=torch.ones(2,28);q[1]=4
    result=distances(source,q,torch.ones(28),torch.full((28,),2.))
    torch.testing.assert_close(result,torch.tensor([.5,0.],dtype=torch.float64),atol=1e-12,rtol=0)
def test_predeclared_classification_boundaries():
    assert classify(.7,-.3,1,2)=='promising source-support proxy'
    assert classify(.7,-.3,2,1)=='weak/mixed support relation'
    assert classify(.59,-.2,1,2)=='weak/mixed support relation'
    assert classify(.59,-.19,1,2)=='source-support hypothesis not supported'

def test_external_freeze_binding(tmp_path):
    import hashlib,json
    from barrier import verify_freeze
    support=tmp_path/'support';support.mkdir()
    freeze=support/'freeze.json';freeze.write_bytes(b'original freeze')
    deployment=tmp_path/'deployment.json'
    expected=hashlib.sha256(freeze.read_bytes()).hexdigest()
    deployment.write_text(json.dumps({'support_freeze_sha256':expected}))
    assert verify_freeze(support,deployment)==expected
    freeze.write_bytes(b'replaced complete support directory')
    with pytest.raises(AssertionError,match='differs from deployment'):
        verify_freeze(support,deployment)

def test_reference_checks_freeze_before_image_opener():
    import ast
    script=Path(__file__).resolve().parents[1]/'research_log/T031A_support/reference.py'
    tree=ast.parse(script.read_text())
    check=next(n.lineno for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='verify_freeze')
    opener=next(n.lineno for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and t.value.id=='Image' and t.attr=='open' for t in n.targets))
    assert check<opener
