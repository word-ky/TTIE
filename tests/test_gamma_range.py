import ast
import inspect
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from PIL import Image
from ttie.ev_range_box import DarkEV2Box
from ttie.gamma_range_box import Gamma05Box
from ttie.semantic_ttt import Region2
from ttie import ev_range_ttt,gamma_range_ttt,lolv2_ev_core,lolv2_gamma_core
from ttie.lolv2_gamma_core import low_image_opener,native_rgb,save_episode
from ttie.energy_model import EnergyHead
from test_semantic_ttt import scorer,RECEIPT


def test_only_active_gamma_lower_changes_and_endpoint_is_exact():
    obj=SimpleNamespace(active=torch.tensor([True,True,False,True]),winner=torch.tensor([0,1,0,0]))
    old=DarkEV2Box(obj,2);new=Gamma05Box(obj,2)
    expected=old.lower.clone();expected[0,1]=torch.where(obj.active.reshape(2,2),.5,expected[0,1])
    assert torch.equal(new.lower,expected) and torch.equal(new.upper,old.upper)
    assert torch.isneginf(new.raw(new.lower)[0,1,0,0])
    for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
        model=Region2(obj.active).to(device)
        with torch.no_grad():model.raw.fill_(-10.)
        new(model);grid=model.physical_grid()
        assert grid[0,1,0,0].item()==.5 and grid[0,1,0,1].item()==.5
        assert grid[0,1,1,0].item()==1.
        assert grid[0,0,0,0].item()==0. and abs(grid[0,0,0,1].item()+.5)<1e-6
        image=torch.full((1,3,16,18),.1,device=device);output=model(image)
        gradient,=torch.autograd.grad(output.mean(),model.raw)
        assert torch.isfinite(output).all() and torch.isfinite(gradient).all()
        assert torch.equal(output[:,:,8:,:9],image[:,:,8:,:9])


def test_trajectory_and_runner_match_accepted_except_bound_wiring_and_metadata():
    assert ast.dump(ast.parse(inspect.getsource(ev_range_ttt.trajectory)))==ast.dump(ast.parse(inspect.getsource(gamma_range_ttt.trajectory)))
    current=inspect.getsource(lolv2_gamma_core).replace('T026-A','T022-C').replace('from .gamma_range_ttt import trajectory','from .ev_range_ttt import trajectory').replace('T014 Region2 dark EV upper2 active gamma lower0.5','T014 Region2 dark EV upper2').replace(',active_gamma_lower=.5','').replace('task_references_deployed=False','normal_images_available=False')
    # Docstrings/comments are descriptive only; restore the first module docstring.
    assert ast.dump(ast.parse(current))==ast.dump(ast.parse(inspect.getsource(lolv2_ev_core)))


def test_100_target_mutation_and_withholding_do_not_change_hashes():
    torch.manual_seed(7);head=EnergyHead().eval().requires_grad_(False)
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);low=root/'low.png';Image.new('RGB',(16,18),(25,35,45)).save(low)
        normals=root/'normal';normals.mkdir()
        for i in range(100):Image.new('RGB',(16,18),(150,150,150)).save(normals/f'{i}.png')
        receipts=[]
        for mode in ['original','changed','withheld']:
            if mode=='changed':
                for i in range(100):Image.new('RGB',(16,18),(i,255-i,0)).save(normals/f'{i}.png')
            if mode=='withheld':normals.rename(root/'unavailable-normal')
            opened=[]
            with patch('PIL.Image.open',low_image_opener({str(low.resolve())},opened)):
                image=native_rgb(low)
                try:native_rgb(normals/'0.png')
                except PermissionError:pass
                else:raise AssertionError('Reference decoder was not denied')
                result,t,decision=gamma_range_ttt.trajectory(image,scorer(),RECEIPT,head,max_steps=40)
                receipts.append(save_episode(root/mode,result,t,decision))
            assert opened==[str(low.resolve())]
            restored=torch.load(root/mode/'output.pt',weights_only=True)
            assert all(torch.equal(restored[k],result[k]) for k in restored)
            assert restored['image'].untyped_storage().nbytes()==restored['image'].numel()*restored['image'].element_size()
        assert receipts[0]==receipts[1]==receipts[2]
