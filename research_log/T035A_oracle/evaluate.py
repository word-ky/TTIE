"""REFERENCE_ORACLE_ONLY: score frozen two-start WB outputs against read-only T028."""
from core import *
import argparse,json,csv,math
import numpy as np
from collections import Counter
from scripts.evaluate_t026a import independent_ssim
o=prior
p=argparse.ArgumentParser()
for k in ['split','normal-root','prior-oracle','preflight','full-wb','out']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();assert o.sha(a.split)==o.SPLIT_SHA
pre=json.loads((a.preflight/'preflight.json').read_bytes());assert o.sha(a.prior_oracle/'per_image.csv')==pre['prior_csv_sha256']
assert o.sha(a.prior_oracle/'freeze.json')==pre['prior_freeze_sha256']
assert o.sha(a.full_wb/'freeze.json')==pre['full_wb_freeze_sha256'] and o.sha(a.full_wb/'per_image.csv')==pre['full_wb_csv_sha256']
full=list(csv.DictReader((a.full_wb/'per_image.csv').open()))
old=list(csv.DictReader((a.prior_oracle/'per_image.csv').open()));split=json.loads(a.split.read_bytes())['selected'];f=json.loads((a.out/'freeze.json').read_bytes())
assert len(f['rows'])==len(split)==len(old)==100 and f['updates']==100000
allowed={str((a.normal_root/r['normal']).resolve()) for r in split};opened=[];original=o.Image.open
def normal_only(path,*args,**kwargs):
    name=str(Path(path).resolve());assert name in allowed;opened.append(name);return original(path,*args,**kwargs)
o.Image.open=normal_only;torch.set_num_threads(1);rows=[];error=0.;bounds=[]
for i,(r,baseline,binding) in enumerate(zip(split,old,f['rows'])):
    assert r['low']==baseline['low']==binding['low']==full[i]['low'];d=a.out/f'{i:03d}'
    for n,h in binding['files'].items():assert o.sha(d/n)==h
    assert o.sha(a.normal_root/r['normal'])==r['normal_sha256']
    target=o.native(a.normal_root/r['normal']);saved=torch.load(d/'oracle_output.pt',weights_only=True,map_location='cpu')
    outputs=torch.load(d/'start_outputs.pt',weights_only=True,map_location='cpu')['images']
    state=json.loads((d/'oracle_states.json').read_bytes());hist=torch.load(d/'histories.pt',weights_only=True,map_location='cpu')['starts'];best_steps=[]
    lo=torch.tensor(state['box']['lower']);hi=torch.tensor(state['box']['upper']);active=torch.tensor(state['gate']['active']).reshape(2,2)
    for j,h in enumerate(hist):
        assert h['raw'].shape==(501,1,3,2,2) and h['mse'].shape==(501,) and state['starts'][j]['updates']==500
        assert torch.isfinite(h['raw']).all() and torch.isfinite(h['mse']).all()
        raw=h['raw'][:,0];grid=common_physical(raw)
        assert (grid[:,:2]>=lo-1e-6).all() and (grid[:,:2]<=hi+1e-6).all()
        assert (grid[:,2:5]>=.5).all() and (grid[:,2:5]<=2).all()
        assert torch.equal(grid[:,2:5,: ,:][:,:,~active],torch.ones_like(grid[:,2:5,: ,:][:,:,~active]))
        step=int(h['mse'].argmin());best_steps.append(step)
        assert step==state['starts'][j]['best_step'] and torch.equal(h['raw'][step],torch.tensor(state['starts'][j]['best_raw']))
    winner=min(range(2),key=lambda j:(float(hist[j]['mse'][best_steps[j]]),best_steps[j],j))
    assert state['winner']==['identity','T026A_selected'][winner] and torch.equal(saved['raw'],hist[winner]['raw'][best_steps[winner]])
    assert torch.equal(saved['image'],outputs[winner])
    scores={}
    for name,image in [('identity',outputs[0]),('selected',outputs[1]),('oracle',saved['image'])]:
        assert torch.isfinite(image).all() and 0<=image.min()<=image.max()<=1
        scores[name]=o.score(image,target);x=image[0].permute(1,2,0).numpy().astype(np.float64);y=target[0].permute(1,2,0).numpy().astype(np.float64)
        independent=-10*np.log10(np.dot((x-y).ravel(),(x-y).ravel())/x.size)
        error=max(error,abs(scores[name]['psnr']-independent),abs(scores[name]['ssim']-independent_ssim(x,y)))
    for j,name in enumerate(['identity','selected']):assert abs(scores[name]['mse']-float(hist[j]['mse'][best_steps[j]]))<1e-10
    assert error<1e-11
    grid=saved['grid'];row=dict(index=i,low=r['low'],winner=state['winner'],best_step=state['best_step'],identity_best_step=best_steps[0],selected_best_step=best_steps[1],seconds=binding['seconds'])
    for name,s in scores.items():row.update({name+'_'+k:v for k,v in s.items()})
    row.update(t028_psnr=float(baseline['oracle_psnr']),t028_ssim=float(baseline['oracle_ssim']),delta_psnr=scores['oracle']['psnr']-float(baseline['oracle_psnr']),delta_ssim=scores['oracle']['ssim']-float(baseline['oracle_ssim']))
    row.update(full_wb_psnr=float(full[i]['oracle_psnr']),full_wb_ssim=float(full[i]['oracle_ssim']),wb_minus_common_psnr=float(full[i]['oracle_psnr'])-scores['oracle']['psnr'],wb_minus_common_ssim=float(full[i]['oracle_ssim'])-scores['oracle']['ssim'])
    assert torch.equal(grid[:,2],grid[:,3]) and torch.equal(grid[:,3],grid[:,4])
    for ch,name in enumerate(['ev','gamma','common_gain']):
        for y in range(2):
            for x in range(2):
                value=float(grid[0,ch,y,x]);lower=float(lo[0,ch,y,x]) if ch<2 else .5;upper=float(hi[0,ch,y,x]) if ch<2 else 2.
                bounds.append(dict(index=i,channel=name,region=f'{y}{x}',active=bool(active[y,x]),value=value,lower_hit=abs(value-lower)<=1e-6,upper_hit=abs(value-upper)<=1e-6))
                row[f'{name}_{y}{x}']=value
    rows.append(row)
with (a.out/'per_image.csv').open('w',newline='') as out:
    w=csv.DictWriter(out,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def stats(v):return dict(count=len(v),mean=float(np.mean(v)),median=float(np.median(v)),min=float(min(v)),max=float(max(v)),p05=float(np.quantile(v,.05)),p95=float(np.quantile(v,.95))) if v else dict(count=0)
metrics={k:stats([r[k] for r in rows]) for k in ['identity_psnr','identity_ssim','selected_psnr','selected_ssim','oracle_psnr','oracle_ssim','t028_psnr','t028_ssim','delta_psnr','delta_ssim','full_wb_psnr','full_wb_ssim','wb_minus_common_psnr','wb_minus_common_ssim']}
mean=metrics['delta_psnr']['mean'];median=metrics['delta_psnr']['median']
classification='common-mode explains most WB gain' if mean>=1.835968277 and median>=1.431879733 else ('chromatic degrees essential' if mean<.5 else 'mixed attribution')
distributions={}
for name in ['ev','gamma','common_gain']:
    distributions[name]={}
    for region in ['00','01','10','11','all']:
        distributions[name][region]={}
        for group in ['active','inactive','all']:
            chosen=[r for r in bounds if r['channel']==name and (region=='all' or r['region']==region) and (group=='all' or r['active']==(group=='active'))]
            distributions[name][region][group]=dict(**stats([r['value'] for r in chosen]),lower_hits=sum(r['lower_hit'] for r in chosen),upper_hits=sum(r['upper_hit'] for r in chosen))
summary=dict(classification=classification,metrics=metrics,histograms={k:dict(Counter(str(r[k]) for r in rows)) for k in ['winner','best_step','identity_best_step','selected_best_step']},
    runtime=stats([r['seconds'] for r in rows]),bounds_and_gains=distributions,win_equal_loss={m:dict(win=sum(r['delta_'+m]>0 for r in rows),equal=sum(r['delta_'+m]==0 for r in rows),loss=sum(r['delta_'+m]<0 for r in rows)) for m in ['psnr','ssim']})
o.write(a.out/'summary.json',summary);o.write(a.out/'bound_values.json',bounds)
o.write(a.out/'evaluation_receipt.json',dict(label=o.LABEL,completed_utc=o.utc(),freeze_sha256=o.sha(a.out/'freeze.json'),all200_histories_finite_bounded=True,all_earliest_winners_verified=True,
    independent_metric_max_abs_error=error,opened_normals=opened,prior_csv_sha256=o.sha(a.prior_oracle/'per_image.csv'),full_wb_csv_sha256=o.sha(a.full_wb/'per_image.csv'),official_test_access=False))
assert len(opened)==100
print(classification,metrics['delta_psnr'],flush=True)
