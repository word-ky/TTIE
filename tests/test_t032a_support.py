import inspect
import pytest
import torch
from ttie.support_trust import cross_image_distances,select,bind_radius


def test_cross_image_excludes_same_id_and_standardizes():
    f=torch.tensor([0.,.1,4.,6.])[:,None].repeat(1,28)
    ids=torch.tensor([0,0,1,1])
    d=cross_image_distances(f,ids,torch.ones(28),torch.full((28,),2.))
    expected=torch.tensor([2.,1.95,1.95,2.95],dtype=torch.float64)
    torch.testing.assert_close(d,expected,atol=1e-7,rtol=0)


def test_first_exit_boundaries_and_ties():
    assert list(inspect.signature(select).parameters)==['energies','distances','radius']
    assert select([3,2,1],[1,1,1],1)['selected_step']==2
    assert select([3,2,1],[1.01,.1,.1],1)['selected_step']==0
    r=select([3,2,0,-1],[.1,.9,1.1,.2],1)
    assert r==dict(original_step=3,selected_step=1,cutoff=1,exit_step=2)
    assert select([2,1,1],[0,0,0],1)['selected_step']==1
    assert select([3],[2],1)['selected_step']==0


def test_radius_mismatch_rejected():
    receipt={'r_support':1.}
    spec={'radius_freeze_sha256':'abc','r_support':1.,'percentile':.95,'quantile_method':'linear'}
    assert bind_radius(receipt,spec,'abc')==1.
    with pytest.raises(AssertionError,match='freeze mismatch'):bind_radius(receipt,spec,'changed')
    with pytest.raises(AssertionError,match='value mismatch'):bind_radius(receipt,dict(spec,r_support=2.),'abc')


def test_support_selects_from_unchanged_baseline_trajectory():
    from ttie.gamma_range_ttt import trajectory
    from ttie.energy_model import EnergyHead
    from test_semantic_ttt import scorer,RECEIPT
    torch.manual_seed(7)
    image=torch.full((1,3,16,16),.1)
    original,t,decision=trajectory(image,scorer(),RECEIPT,EnergyHead().eval().requires_grad_(False),max_steps=2)
    energies=decision['scores'];states=t['states'].clone();images=t['images'].clone()
    chosen=select(energies,[0.]*len(energies),1.)
    assert chosen['selected_step']==decision['selected_step']
    assert torch.equal(original['image'],t['images'][chosen['selected_step']])
    select(energies,[2.]*len(energies),1.)
    assert torch.equal(states,t['states']) and torch.equal(images,t['images'])
