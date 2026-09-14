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
