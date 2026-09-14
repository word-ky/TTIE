import inspect
import torch
from ttie.self_reversal import select
from ttie.lolv2_gamma_core import low_image_opener

def setup():return list(reversed(range(41))),torch.ones(41,1,2,2,2),torch.tensor([True,False,False,False])
def test_first_crossing_prefix_and_tie():
    e,g,a=setup();g[15,0,:,0,0]=-1;g[:,:, :,1,1]=-999
    d=select(e,g,a);assert d['crossing']==15 and d['cutoff']==14 and d['selected_step']==14 and d['original_step']==40
    e[3]=e[8]=-100;assert select(e,g,a)['selected_step']==3
def test_no_crossing_zero_and_degenerate_fallback():
    e,g,a=setup();assert select(e,g,a)['selected_step']==40
    g[12]=0;d=select(e,g,a);assert d['fallback']=='degenerate_comparison' and d['selected_step']==40
    g[10]=0;assert select(e,g,a)['fallback']=='degenerate_anchor'
    e,g,a=setup();g[11,0,1,0,0]=-1
    assert select(e,g,a)['crossing']==11
def test_selector_no_reference_arguments_or_decode_path(tmp_path,monkeypatch):
    assert list(inspect.signature(select).parameters)==['energies','gradients','active']
    from PIL import Image
    def forbidden(*args,**kwargs):raise AssertionError('selector must not decode')
    monkeypatch.setattr(Image,'open',forbidden)
    e,g,a=setup();assert select(e,g,a)['selected_step']==40
    normal=tmp_path/'normal.png';normal.write_bytes(b'changed target')
    assert select(e,g,a)['selected_step']==40
    guarded=low_image_opener({str(tmp_path/'low.png')},[])
    import pytest
    with pytest.raises(PermissionError):guarded(normal)
