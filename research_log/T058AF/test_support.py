import json
import pytest
import torch
from PIL import Image
from research_log.T058AF.support import SourceOpens,reference_loss,save_reference,replay,atomic_json,sha,thash,utc,alignment,summary
from research_log.T058A_tangent.core import Detail

def test_source_open_order_hash_and_allowlist(tmp_path):
    image=tmp_path/'source.png';Image.new('RGB',(4,4)).save(image)
    sources={1:dict(image_id=1,filename=image.name,split='train_t014_sobolev',sha256=sha(image))}
    gate=SourceOpens(tmp_path,sources)
    with pytest.raises(AssertionError,match='STAGE_A_NOT_VERIFIED'):gate(image)
    gate.verified_utc=utc()
    with pytest.raises(AssertionError,match='UNAUTHORIZED_IMAGE'):gate(tmp_path/'other.png')
    with gate(image) as im:assert im.size==(4,4)
    assert len(gate.opened)==1 and gate.opened[0]['utc']>gate.verified_utc
    with pytest.raises(AssertionError,match='DUPLICATE'):gate(image)
    image.write_bytes(b'changed')
    with pytest.raises(AssertionError,match='HASH'):gate(image)

def test_reference_scalar_and_zero_identity():
    torch.manual_seed(4);y=torch.rand(1,3,10,12);clean=torch.rand_like(y)
    model=Detail(y,torch.ones(4,dtype=torch.bool),torch.zeros(1,2,2,2))
    loss=reference_loss(model(),clean);assert loss.dtype==torch.float64
    assert torch.equal(loss,(y.double()-clean.double()).square().mean())
    g,=torch.autograd.grad(loss,model.v);assert g.shape==(1,1,8,8) and torch.isfinite(g).all() and model.v.grad is None
    inactive=Detail(y,torch.zeros(4,dtype=torch.bool),torch.zeros(1,2,2,2))
    zero,=torch.autograd.grad(reference_loss(inactive(),clean),inactive.v);assert torch.count_nonzero(zero)==0

def test_persist_replay_zero_and_opposing_gradients(tmp_path):
    e=[torch.ones(1,1,8,8) for _ in range(3)];r=[e[0].clone(),-e[1],torch.zeros_like(e[2])]
    rows=[]
    for i,(a,b) in enumerate(zip(e,r)):
        rows.append(dict(index=i,energy_sha256=thash(a),reference_sha256=thash(b),condition='x',image_id=1,state_name='identity',**alignment(a,b,torch.ones_like(a,dtype=torch.bool))))
    chunk=save_reference(tmp_path,[0,1,2],r);atomic_json(tmp_path/'manifest.json',dict(rows=rows,chunks=[chunk]))
    assert replay(tmp_path,e,3)['rows']==3
    s=summary(rows);assert s['overall']['degenerate']==1 and s['overall']['positive_dot_fraction']==.5 and s['overall']['cosine_median']==0
    assert s['classification']=='frozen-energy detail tangent not ready'
    raw=(tmp_path/chunk['file']).read_bytes();(tmp_path/chunk['file']).write_bytes(raw+b'changed')
    with pytest.raises(AssertionError):replay(tmp_path,e,3)

def test_frozen_gate_boundaries():
    from research_log.T058A_tangent.core import classify
    assert classify(dict(nondegenerate=1,positive_dot_fraction=.75,cosine_median=.5))=='frozen-energy detail tangent supported on source'
    for key in ['positive_dot_fraction','cosine_median']:
        s=dict(nondegenerate=1,positive_dot_fraction=.75,cosine_median=.5);s[key]-=1e-8
        assert classify(s)=='frozen-energy detail tangent not ready'
