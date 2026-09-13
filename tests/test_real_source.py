import ast
import inspect
import json
import hashlib
import tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from PIL import Image
from ttie import ev_range_ttt,lolv2_ev_core,lolv2_real_core
from ttie.lolv2_real_core import low_image_opener,native_rgb,save_episode
from ttie.energy_model import EnergyHead
from ttie.semantic_ttt import FixedObjective,Region2
from ttie.sobolev_source import derivative_record
from ttie.sobolev_train import raw_gradient,train_pair
from ttie.real_source_sobolev import train_primary
from test_semantic_ttt import scorer,RECEIPT


def test_source_selection_is_fixed_and_disjoint():
    allrows=json.loads(Path('research_log/T022A_data/dataset_manifest.json').read_bytes())['records']
    val=json.loads(Path('research_log/T022A_data/split.json').read_bytes())['selected']
    names={r['low'] for r in val}
    eligible=[r for r in allrows if r['low'].startswith('Train/') and r['low'] not in names]
    ordered=sorted(eligible,key=lambda r:(hashlib.sha256(r['low'].encode()).hexdigest(),r['low']))[:16]
    saved=json.loads(Path('research_log/T023A_source_manifest.json').read_bytes())
    assert len(eligible)==589 and saved['selected']==ordered


def test_validation_runner_unchanged_except_task_label():
    old=inspect.getsource(lolv2_ev_core)
    new=inspect.getsource(lolv2_real_core).replace('T023-A','T022-C')
    assert ast.dump(ast.parse(old))==ast.dump(ast.parse(new))


def test_ev2_derivative_matches_direct_chain_rule_beyond_old_cap():
    for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
        image=torch.full((1,3,16,18),.1,device=device);clean=image+.25
        obj=FixedObjective(scorer().to(device),image,RECEIPT);model=Region2(obj.active).to(image)
        with torch.no_grad():model.raw[:,0].fill_(torch.atanh(torch.tensor(.6)).item())
        assert model.physical_grid()[0,0].min()>.5
        record=derivative_record(image,clean,obj,model.raw)
        head=EnergyHead().to(device).eval().requires_grad_(False)
        energy,out,_,_,_=ev_range_ttt.evaluate_energy(model,image,obj,head)
        direct=torch.autograd.grad(energy,model.raw,retain_graph=True)[0].flatten().cpu()
        truth=torch.autograd.grad(((out-clean).square().mean()+1e-6).log(),model.raw)[0].flatten().cpu()
        head=head.cpu();x=record['features'][None].requires_grad_()
        cached=raw_gradient(head,head.standardized(x),x,record['jacobian'][None])[0]
        torch.testing.assert_close(cached,direct,atol=2e-7,rtol=2e-5)
        torch.testing.assert_close(record['reference_gradient'],truth,atol=0,rtol=0)
        assert torch.equal(record['jacobian'][:12],torch.zeros(12,8))


def test_single_head_is_exact_original_primary_recipe():
    torch.manual_seed(13);x=torch.randn(16,28);mse=torch.linspace(.002,.07,16)
    records=dict(jacobian=torch.randn(16,28,8),reference_gradient=torch.randn(16,8),direction_mask=torch.ones(16,dtype=torch.bool))
    heads,histories=train_pair(x,mse,records)
    actual,history=train_primary(x,mse,records)
    assert history==histories['sobolev_primary'] and len(history)==100
    assert all(torch.equal(v,heads['sobolev_primary'].state_dict()[k]) for k,v in actual.state_dict().items())


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
                result,t,decision=ev_range_ttt.trajectory(image,scorer(),RECEIPT,head,max_steps=40)
                receipts.append(save_episode(root/mode,result,t,decision))
            assert opened==[str(low.resolve())]
            restored=torch.load(root/mode/'output.pt',weights_only=True)
            assert all(torch.equal(restored[k],result[k]) for k in restored)
            assert restored['image'].untyped_storage().nbytes()==restored['image'].numel()*restored['image'].element_size()
        assert receipts[0]==receipts[1]==receipts[2]
