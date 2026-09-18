import numpy as np
from research_log.T059V.core import errors,passed

def test_zero_equality():
    r=errors(np.zeros(64),np.zeros(64));assert r['relative']==0 and r['cosine']==1

def test_fixed_bounds():
    row=dict(x=dict(max_abs=5e-5),J=dict(relative=2e-3),g=dict(cosine=.999,relative=1e-2),y1=dict(max_abs=1e-4));assert passed(row)
    row['x']['max_abs']=5.001e-5;assert not passed(row)

def test_signed_gradient_error():
    r=errors(np.array([1.,0]),np.array([-1.,0]));assert r['cosine']==-1 and r['relative']==2
