import ast
import inspect
import tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from PIL import Image
from ttie import gamma_range_ttt,lolv2_gamma_core,lolv2_gamma_budget_core
from ttie.lolv2_gamma_budget_core import low_image_opener,native_rgb,save_episode
from ttie.energy_model import EnergyHead
from test_semantic_ttt import scorer,RECEIPT


def test_only_declared_budget_changes():
    old=inspect.getsource(lolv2_gamma_core)
    new=inspect.getsource(lolv2_gamma_budget_core).replace('T026-B','T026-A').replace('max_steps=80','max_steps=40')
    assert ast.dump(ast.parse(old))==ast.dump(ast.parse(new))


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
                result,t,decision=gamma_range_ttt.trajectory(image,scorer(),RECEIPT,head,max_steps=80)
                assert len(t['states'])==81 and t['diagnostics']['steps']==80
                receipts.append(save_episode(root/mode,result,t,decision))
            assert opened==[str(low.resolve())]
            restored=torch.load(root/mode/'output.pt',weights_only=True)
            assert all(torch.equal(restored[k],result[k]) for k in restored)
            assert restored['image'].untyped_storage().nbytes()==restored['image'].numel()*restored['image'].element_size()
        assert receipts[0]==receipts[1]==receipts[2]


def test_metric_arithmetic_unchanged():
    base=Path('scripts/evaluate_t026a.py').read_text()
    new=Path('scripts/evaluate_t026b.py').read_text().replace("(80 if any(d['gate']['active']) else 0)","(40 if any(d['gate']['active']) else 0)")
    assert ast.dump(ast.parse(base))==ast.dump(ast.parse(new.replace('T026B','T026A')))
