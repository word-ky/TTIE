import numpy as np
from research_log.T059T.core import diagnose,classify

def test_first_order_and_overshoot():
    g=np.ones(64)*.01;v=-.05*g/(abs(g)+1e-8);ref=np.ones(64)*1e-5
    d=diagnose(g,v,ref,.1,.1001)
    assert d['L']<0 and d['A']>0 and d['R']>d['A'] and d['overshoot_flip']
    assert d['saturation_fraction']==1 and d['adam_replay_max_error']==0

def test_saturation_excludes_small_coordinates():
    g=np.ones(64)*1e-12;g[0]=1e-6;v=-.05*g/(abs(g)+1e-8)
    d=diagnose(g,v,-np.ones(64),.1,.09)
    assert d['coordinates_above_eps']==1 and d['saturation_fraction']==1
    assert not d['linear_descent'] and not d['overshoot_flip']

def test_ordered_exact_gates():
    assert classify(89,100,4,4,1).startswith('transferred')
    assert classify(90,100,1,4,1).startswith('optimizer-scale/curvature mismatch not')
    assert classify(90,100,2,4,.79).startswith('optimizer-scale/curvature mismatch not')
    assert classify(90,100,2,4,.80).startswith('optimizer-scale/curvature mismatch is supported')
