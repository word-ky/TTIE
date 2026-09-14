import ast,inspect
from types import SimpleNamespace
import torch
from ttie import gamma_range_ttt,common_gain_ttt
from ttie.common_gain import CommonRegion2,CommonBox,physical_grid
from ttie.semantic_ttt import Region2
from ttie.gamma_range_box import Gamma05Box
from ttie.energy_model import EnergyHead
from test_semantic_ttt import scorer,RECEIPT

def test_exact_trajectory_energy_and_low_only_api():
    for name in ['trajectory','evaluate_energy']:
        assert ast.dump(ast.parse(inspect.getsource(getattr(common_gain_ttt,name))))==ast.dump(ast.parse(inspect.getsource(getattr(gamma_range_ttt,name))))
    assert list(inspect.signature(common_gain_ttt.trajectory).parameters)==['image','scorer','receipt','head','basis','max_steps']

def test_identity_renderer_inactive_gain_and_one_coordinate():
    torch.manual_seed(7);image=torch.rand(1,3,13,17)
    obj=SimpleNamespace(active=torch.tensor([True,False,True,True]),winner=torch.tensor([0,0,1,0]))
    old=Region2(obj.active);new=CommonRegion2(obj.active);oldbox=Gamma05Box(obj,2);box=CommonBox(obj,2)
    with torch.no_grad():old.raw.uniform_(-.4,.4);oldbox(old);new.raw[:,:2].copy_(old.raw)
    assert (old(image)-new(image)).abs().max()<=1e-6
    with torch.no_grad():new.raw[:,2:].fill_(.5);box(new)
    assert new.raw.numel()==12 and torch.equal(new(image)[:,:,:6,8:],image[:,:,:6,8:])
    assert torch.equal(new.physical_grid()[:,2:5,0,1],torch.ones(1,3))

def test_frozen_energy_selection_and_bounded_states():
    torch.manual_seed(7);head=EnergyHead().eval().requires_grad_(False);image=torch.full((1,3,16,18),.1)
    result,t,d=common_gain_ttt.trajectory(image,scorer(),RECEIPT,head,max_steps=40)
    assert t['diagnostics']['steps']==(40 if any(t['gate']['active']) else 0)
    assert d['selected_step']==min(range(len(d['scores'])),key=lambda i:(d['scores'][i],i))
    assert torch.equal(result['raw'],t['states'][d['selected_step']])
    grid=physical_grid(t['states'][:,0]);assert torch.isfinite(t['states']).all()
    assert (grid[:,2:5]>=.5).all() and (grid[:,2:5]<=2).all()
    assert torch.equal(grid[:,2],grid[:,3]) and torch.equal(grid[:,3],grid[:,4])
