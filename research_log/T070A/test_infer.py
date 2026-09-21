import inspect,json
from pathlib import Path
import numpy as np
from research_log.T070A.infer import FinalOurs,select_trajectory
from research_log.T067B.core import choices
from research_log.T063C.core import choose

def test_degraded_image_only_api():
    assert list(inspect.signature(FinalOurs.__call__).parameters)==['self','low']
    assert list(inspect.signature(FinalOurs.__init__).parameters)==['self','manifest']
    forbidden={'normal','clean','labels','psnr','ssim','baseline','oracle','condition_id','image_id'}
    assert not forbidden.intersection(inspect.signature(FinalOurs.__call__).parameters)

def test_existing_selection_composition(monkeypatch):
    import research_log.T070A.infer as m
    vals=np.linspace(1,0,28).tolist();probs=np.zeros(28);probs[4:]=.9
    monkeypatch.setattr(m,'features',lambda *a:np.zeros((28,19)))
    monkeypatch.setattr(m,'predict',lambda *a:probs)
    row,table,p=select_trajectory(None,dict(images=[],states=[],values=vals),{})
    expected=next(c for c in choices(vals,probs.tolist(),choose(vals,m.RHO)) if c['lambda_value']==.875)
    assert row==expected and row['k_FS']==4 and table.shape==(28,19)

def test_constants_match_accepted_method():
    c=json.loads(Path('research_log/T070A/constants.json').read_bytes())
    assert c['updates']==27 and c['loss_weights']==[1,10,5] and c['optimizer']['lr']==.03
    assert c['rho']==.9857470621423519 and c['lambda_value']==.875 and c['probability_threshold']==.5
