"""REFERENCE_ORACLE_ONLY: isolated T025-A validation diagnostic, never deployable TTT."""
import argparse
import csv
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from ttie.semantic_ttt import Region2
from ttie.ev_range_box import DarkEV2Box
from ttie.ssim_transfer import rgb_ssim

LABEL = 'REFERENCE_ORACLE_ONLY'
SPLIT_SHA = 'b88c8347005984b5523b117b52c0c068672fe172eb7c9aa5b60102d350e2d85b'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value): Path(path).write_text(json.dumps(value, indent=2)+'\n')
def utc(): return datetime.now(timezone.utc).isoformat()
def native(path):
    with Image.open(path) as im: a=np.asarray(im.convert('RGB')).copy()
    assert a.shape == (400,600,3)
    return torch.from_numpy(a).permute(2,0,1).unsqueeze(0).float()/255.
def mse(output, target): return (output.double()-target.double()).square().mean()
def score(output, target):
    a=output.detach().cpu().squeeze(0).permute(1,2,0).numpy()
    b=target.detach().cpu().squeeze(0).permute(1,2,0).numpy()
    value=float(np.mean((a.astype(np.float64)-b.astype(np.float64))**2))
    return dict(mse=value,psnr=-10*math.log10(value),ssim=rgb_ssim(a,b))
def frozen_model(decision, image):
    obj=SimpleNamespace(active=torch.tensor(decision['gate']['active'],device=image.device,dtype=torch.bool),
                        winner=torch.tensor(decision['gate']['winner'],device=image.device))
    model=Region2(obj.active).to(image); box=DarkEV2Box(obj,2)
    saved=decision['diagnostics']['action_box']
    assert torch.equal(box.lower.cpu(),torch.tensor(saved['lower']))
    assert torch.equal(box.upper.cpu(),torch.tensor(saved['upper']))
    return model,box

def optimize_start(model, box, low, reference, initial, steps=500):
    """Exactly steps Adam updates; retain earliest lowest-MSE state over 0..steps."""
    with torch.no_grad(): model.raw.copy_(initial)
    optimizer=torch.optim.Adam([model.raw],lr=.05)
    history=[]; best=float('inf'); best_step=0; best_raw=None
    for step in range(steps+1):
        output=model(low); loss=mse(output,reference)
        value=float(loss.detach()); assert math.isfinite(value)
        history.append(value)
        if value<best:
            best=value;best_step=step;best_raw=model.raw.detach().clone()
        if step==steps: break
        optimizer.zero_grad(set_to_none=True);loss.backward()
        assert torch.isfinite(model.raw.grad).all()
        optimizer.step();box(model)
    final_raw=model.raw.detach().clone()
    with torch.no_grad(): model.raw.copy_(best_raw)
    return dict(label=LABEL,updates=steps,best_step=best_step,best_mse=best,
                initial_mse=history[0],final_mse=history[-1],history=history,
                best_raw=best_raw.cpu(),final_raw=final_raw.cpu())

def main():
    p=argparse.ArgumentParser(description=LABEL)
    for name in ['split','low-root','normal-root','accepted','out']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args(); assert sha(args.split)==SPLIT_SHA
    split=json.loads(args.split.read_bytes()); assert len(split['selected'])==100
    accepted_freeze=json.loads((args.accepted/'freeze.json').read_bytes())
    accepted_metrics=list(csv.DictReader((args.accepted/'metrics.csv').open()))
    assert len(accepted_freeze['rows'])==len(accepted_metrics)==100
    # Only named validation paths are opened; no traversal/decoding of the official Test directory.
    opened=[];original_open=Image.open
    allowed={str((root/row[key]).resolve()) for row in split['selected'] for root,key in [(args.low_root,'low'),(args.normal_root,'normal')]}
    def open_validation(path,*a,**kw):
        resolved=str(Path(path).resolve());assert resolved in allowed
        opened.append(resolved);return original_open(path,*a,**kw)
    Image.open=open_validation
    torch.manual_seed(7);torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device='cuda:0';assert torch.cuda.is_available()
    args.out.mkdir(parents=True,exist_ok=False)
    code={str(p):sha(p) for p in [Path(__file__),Path('ttie/semantic_ttt.py'),Path('ttie/isp.py'),Path('ttie/ev_range_box.py'),Path('ttie/projected_ttt.py'),Path('ttie/ssim_transfer.py')]}
    config=dict(label=LABEL,task='T025-A',seed=7,optimizer='Adam',lr=.05,betas=[.9,.999],eps=1e-8,weight_decay=0,updates_per_start=500,starts=['identity','T022C_selected'],selection='minimum full-frame RGB MSE; earliest step tie; identity start wins exact inter-start tie',objective='full-frame RGB MSE, float64 accumulation and gradients through unchanged float32 renderer',shape=[1,3,400,600],split_sha256=sha(args.split),accepted_freeze_sha256=sha(args.accepted/'freeze.json'),accepted_metrics_sha256=sha(args.accepted/'metrics.csv'),code_sha256=code,gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,tf32=False,created_utc=utc(),bounds='reuse saved T022-C gate/action box; darkEV0..2 brightEV-.5..0 gamma.8..1.25 inactiveidentity',saturation_tolerance=1e-6,output_tolerance=1e-6,metric_tolerance=1e-6,mse_nonworse_tolerance=1e-10)
    write(args.out/'config.json',config);rows=[];provenance=[]
    for i,row in enumerate(split['selected']):
        start=time.perf_counter();d=args.accepted/f'{i:03d}';frozen=accepted_freeze['rows'][i]
        assert frozen['low']==row['low']==accepted_metrics[i]['low']
        hashes={n:sha(d/n) for n in ['decision.json','output.pt','trajectory.pt']}
        assert all(hashes[n]==frozen['files'][n]['sha256'] for n in hashes)
        assert sha(args.low_root/row['low'])==row['low_sha256']
        assert sha(args.normal_root/row['normal'])==row['normal_sha256']
        decision=json.loads((d/'decision.json').read_bytes());saved=torch.load(d/'output.pt',map_location='cpu',weights_only=True)
        low=native(args.low_root/row['low']).to(device);target=native(args.normal_root/row['normal']).to(device)
        model,box=frozen_model(decision,low)
        with torch.no_grad():
            model.raw.copy_(saved['raw'].to(device));reproduced=model(low)
            error=float((reproduced.cpu()-saved['image']).abs().max())
            grid_error=float((model.physical_grid()[:,:2].cpu()-saved['grid']).abs().max())
        assert error<=1e-6 and grid_error<=1e-6
        raw_score=score(low,target);selected_score=score(saved['image'],target);repro_score=score(reproduced,target)
        for metric in ['psnr','ssim']:
            assert abs(selected_score[metric]-float(accepted_metrics[i]['ours_'+metric]))<=1e-6
            assert abs(repro_score[metric]-selected_score[metric])<=1e-6
            assert abs(raw_score[metric]-float(accepted_metrics[i]['raw_'+metric]))<=1e-6
        results=[]
        for initial in [torch.zeros_like(model.raw),saved['raw'].to(device)]:
            results.append(optimize_start(model,box,low,target,initial))
        winner=min(range(2),key=lambda j:(results[j]['best_mse'],j));best=results[winner]
        with torch.no_grad():
            model.raw.copy_(best['best_raw'].to(device));output=model(low);grid=model.physical_grid()[:,:2]
        oracle_score=score(output,target)
        assert oracle_score['mse']<=selected_score['mse']+1e-10
        assert all(math.isfinite(v) for s in [raw_score,selected_score,oracle_score] for v in s.values())
        assert torch.isfinite(output).all() and torch.isfinite(grid).all() and torch.isfinite(model.raw).all()
        assert bool((grid>=box.lower-1e-6).all() and (grid<=box.upper+1e-6).all())
        active=model.active; sat={}
        for ch,name in enumerate(['ev','gamma']):
            lo=(grid[0,ch]-box.lower[0,ch]).abs()<=1e-6;hi=(grid[0,ch]-box.upper[0,ch]).abs()<=1e-6
            for suffix,mask in [('lower',lo),('upper',hi),('either',lo|hi)]:
                sat[name+'_'+suffix+'_active_count']=int((mask&active).sum())
                sat[name+'_'+suffix+'_all_count']=int(mask.sum())
        dest=args.out/f'{i:03d}';dest.mkdir()
        torch.save(dict(label=LABEL,image=output.detach().cpu(),raw=model.raw.detach().cpu(),grid=grid.detach().cpu()),dest/'oracle_output.pt')
        states=dict(label=LABEL,index=i,low=row['low'],gate=decision['gate'],box=decision['diagnostics']['action_box'],winner=['identity','T022C_selected'][winner],raw=model.raw.detach().cpu().tolist(),physical_grid=grid.detach().cpu().tolist(),starts=[{k:(v.tolist() if isinstance(v,torch.Tensor) else v) for k,v in a.items()} for a in results])
        write(dest/'oracle_states.json',states)
        record=dict(label=LABEL,index=i,low=row['low'],normal=row['normal'],winner=states['winner'],best_step=best['best_step'],identity_best_step=results[0]['best_step'],selected_best_step=results[1]['best_step'],identity_final_mse=results[0]['final_mse'],selected_final_mse=results[1]['final_mse'],identity_updates=500,selected_updates=500,active_count=int(active.sum()),dark_active_count=int((active.flatten() & (torch.tensor(decision['gate']['winner'],device=device)==0)).sum()),reproduction_max_abs=error,grid_max_abs=grid_error,seconds=time.perf_counter()-start,**sat)
        for name,s in [('raw',raw_score),('selected',selected_score),('oracle',oracle_score)]:record.update({name+'_'+k:v for k,v in s.items()})
        record.update(delta_psnr=oracle_score['psnr']-selected_score['psnr'],delta_ssim=oracle_score['ssim']-selected_score['ssim'])
        rows.append(record);provenance.append(dict(label=LABEL,index=i,low=row['low'],input_hashes={**hashes,'low':row['low_sha256'],'normal':row['normal_sha256']},oracle_output_sha256=sha(dest/'oracle_output.pt'),oracle_states_sha256=sha(dest/'oracle_states.json')))
        write(args.out/'progress.json',dict(label=LABEL,completed=len(rows),last=record,utc=utc()))
        print(f'{LABEL} {i+1}/100 winner={states["winner"]} step={best["best_step"]} delta_psnr={record["delta_psnr"]:.4f} seconds={record["seconds"]:.2f}',flush=True)
    with (args.out/'per_image.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(args.out/'per_image.json',dict(label=LABEL,rows=rows));write(args.out/'provenance.json',dict(label=LABEL,rows=provenance))
    assert len(opened)==200 and set(opened)==allowed
    assert all(sha(p)==h for p,h in code.items())
    write(args.out/'completion.json',dict(label=LABEL,status='diagnostic-complete',utc=utc(),pairs=100,starts=200,updates=100000,all_finite=True,all_selected_starts_reproduced=True,all_oracle_mse_nonworse=True,official_test_opened=False,opened_validation_images=opened,config_sha256=sha(args.out/'config.json'),provenance_sha256=sha(args.out/'provenance.json'),per_image_sha256=sha(args.out/'per_image.csv')))
    print('REFERENCE_ORACLE_ONLY all100 complete; no deployable state changed',flush=True)

if __name__=='__main__':main()
