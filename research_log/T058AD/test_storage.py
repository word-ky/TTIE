import json
import pytest,torch
from research_log.T058AD.storage import atomic_json,save_chunk,reopen,continuity
from research_log.T058A_tangent.core import thash

def test_checkpoint_roundtrip_and_no_false_completion(tmp_path):
    gs=[torch.zeros(1,1,8,8),torch.ones(1,1,8,8)]
    chunk=save_chunk(tmp_path,[0,1],gs)
    rows=[dict(index=i,gradient_sha256=thash(g),gradient_norm=float(g.double().norm())) for i,g in enumerate(gs)]
    atomic_json(tmp_path/'manifest.json',dict(rows=rows,chunks=[chunk],complete=False))
    assert reopen(tmp_path,2)['rows']==2
    assert not (tmp_path/'complete.json').exists()
    with pytest.raises(AssertionError):reopen(tmp_path,3)
    saved=torch.load(tmp_path/chunk['file'],weights_only=True);saved['g_E'][1,0,0,0,0]+=1;torch.save(saved,tmp_path/chunk['file'])
    with pytest.raises(AssertionError):reopen(tmp_path,2)

def test_interrupted_chunk_not_published(tmp_path,monkeypatch):
    def fail(*args,**kwargs):raise OSError('simulated interrupted write')
    monkeypatch.setattr(torch,'save',fail)
    with pytest.raises(OSError):save_chunk(tmp_path,[0],[torch.zeros(1,1,8,8)])
    assert not (tmp_path/'gradients_0000_0000.pt').exists() and not (tmp_path/'complete.json').exists()

def test_continuity_fixed_thresholds():
    z=torch.zeros(1,1,8,8);g=torch.arange(64,dtype=torch.float32).reshape_as(z)/100
    assert continuity(z,z)['passed'] and continuity(g,g)['passed']
    assert not continuity(-g,g)['passed']
