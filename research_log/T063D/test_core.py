import hashlib
import pytest
from research_log.T063D.prepare import select
from research_log.T063C.core import choose
from research_log.T062CR2.core import trajectory
import torch


def test_cohort_rank_and_exclusion():
    pairs=[dict(low=f'Train/Low/low{i:05d}.png',normal=f'Train/Normal/normal{i:05d}.png') for i in range(150)]
    used={p['normal'] for p in pairs[:20]};selected=select(pairs,used)
    expected=sorted(pairs[20:],key=lambda p:hashlib.sha256(('T063D:'+p['low']).encode()).digest())[:100]
    assert [(p['low'],p['normal']) for p in selected]==[(p['low'],p['normal']) for p in expected]
    with pytest.raises(AssertionError):select(pairs,{p['normal'] for p in pairs[:60]})


def test_frozen_progress_on_exact_27_update_path():
    from types import SimpleNamespace
    gate=SimpleNamespace(active=torch.tensor([True,False,True,True]),winner=torch.tensor([0,0,1,0]))
    torch.set_num_threads(1)
    x=torch.rand(1,3,32,32)*.2
    t=trajectory(x,gate);k=choose(t['values'],0.9857470621423519)
    threshold=t['values'][0]-.9857470621423519*(t['values'][0]-min(t['values']))
    assert k==next(i for i,v in enumerate(t['values']) if v<=threshold)
    assert len(t['gradients'])==27 and len(t['states'])==28
