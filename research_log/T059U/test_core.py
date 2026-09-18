import torch
from research_log.T059U.core import gate,TAU,classify

def test_exact_threshold():
    assert gate(torch.tensor([TAU],dtype=torch.float64))
    assert not gate(torch.tensor([TAU-1e-12],dtype=torch.float64))
    assert not gate(torch.zeros(64))

def test_ordered_gates():
    assert classify(.74,0,None,0,0,0,None)=='low-norm abstention is too indiscriminate'
    assert classify(.75,0,None,1,0,-1,-1)=='low-norm safety mechanism is not testable on this cohort'
    assert classify(.75,5,.79,.9,1,-1,-1)=='fixed low-norm abstention does not transfer'
    assert classify(.75,5,.8,.8,1,-1,-1)=='fixed low-norm abstention is supported as a source-only outer safety candidate'
    assert classify(.75,5,.8,.8,1,0,-1)=='fixed low-norm abstention does not transfer'
