"""REFERENCE_ORACLE_ONLY scoring after all states/outputs are frozen."""
from core import *
from scripts.evaluate_t026a import independent_ssim
from ttie.isp import physical_parameters

p=argparse.ArgumentParser(description=LABEL)
for n in ['split','low-root','normal-root','accepted','preflight','out']:p.add_argument('--'+n,type=Path,required=True)
a=p.parse_args();assert sha(a.split)==SPLIT_SHA
split=json.loads(a.split.read_bytes())['selected'];freeze=json.loads((a.out/'freeze.json').read_bytes())
pre=json.loads((a.preflight/'preflight.json').read_bytes());assert len(freeze['rows'])==len(split)==100
assert sha(a.accepted/'metrics.csv')==pre['accepted_metrics_sha256']
accepted=list(csv.DictReader((a.accepted/'metrics.csv').open()))
allowed={str((root/r[k]).resolve()) for r in split for root,k in [(a.low_root,'low'),(a.normal_root,'normal')]}
opened=[];original=Image.open
def validation_only(path,*args,**kw):
    resolved=str(Path(path).resolve());assert resolved in allowed
    opened.append(resolved);return original(path,*args,**kw)
Image.open=validation_only
torch.set_num_threads(1);rows=[];errors=[]
for i,r in enumerate(split):
    dest=a.out/f'{i:03d}';binding=freeze['rows'][i];assert r['low']==binding['low']
    assert all(sha(dest/n)==h for n,h in binding['files'].items())
    state=json.loads((dest/'oracle_states.json').read_bytes());hist=torch.load(dest/'histories.pt',weights_only=True)['starts']
    low=native(a.low_root/r['low']);target=native(a.normal_root/r['normal'])
    selected=torch.load(a.accepted/f'{i:03d}'/'output.pt',weights_only=True,map_location='cpu')
    saved=torch.load(dest/'oracle_output.pt',weights_only=True,map_location='cpu')
    scores={name:score(image,target) for name,image in [('raw',low),('selected',selected['image']),('oracle',saved['image'])]}
    assert r['low']==accepted[i]['low']
    for name,image in [('raw',low),('selected',selected['image']),('oracle',saved['image'])]:
        x=image[0].permute(1,2,0).numpy().astype(np.float64);y=target[0].permute(1,2,0).numpy().astype(np.float64)
        errors.extend([abs(scores[name]['ssim']-independent_ssim(x,y)),abs(scores[name]['psnr']-float(-10*torch.log10(torch.from_numpy(x-y).square().mean())))])
        for value in scores[name].values():assert math.isfinite(value)
    for metric in ['psnr','ssim']:
        assert abs(scores['selected'][metric]-float(accepted[i]['ours_'+metric]))<1e-11
        assert abs(scores['raw'][metric]-float(accepted[i]['raw_'+metric]))<1e-11
    lo=torch.tensor(state['box']['lower']);hi=torch.tensor(state['box']['upper'])
    best_steps=[]
    for j,h in enumerate(hist):
        assert h['raw'].shape==(501,1,2,2,2) and h['mse'].shape==(501,)
        assert torch.isfinite(h['raw']).all() and torch.isfinite(h['mse']).all()
        raw=h['raw'][:,0];grid=physical_parameters(torch.cat([raw,torch.zeros(501,4,2,2)],dim=1))[:,:2]
        assert (grid>=lo-1e-6).all() and (grid<=hi+1e-6).all()
        step=int(h['mse'].argmin());best_steps.append(step)
        assert step==state['starts'][j]['best_step'] and torch.equal(h['raw'][step],torch.tensor(state['starts'][j]['best_raw']))
        assert state['starts'][j]['updates']==500
    winner=min(range(2),key=lambda j:(float(hist[j]['mse'][best_steps[j]]),best_steps[j],j))
    assert state['winner']==['identity','T026A_selected'][winner]
    assert torch.equal(saved['raw'],hist[winner]['raw'][best_steps[winner]])
    assert abs(scores['oracle']['mse']-float(hist[winner]['mse'][best_steps[winner]]))<1e-10
    assert scores['oracle']['mse']<=scores['selected']['mse']+1e-10
    active=torch.tensor(state['gate']['active']).reshape(2,2);grid=saved['grid'];sat={}
    for ch,name in enumerate(['ev','gamma']):
        lower=(grid[0,ch]-lo[0,ch]).abs()<=1e-6;upper=(grid[0,ch]-hi[0,ch]).abs()<=1e-6
        for suffix,mask in [('lower',lower),('upper',upper),('either',lower|upper)]:
            for group,selection in [('active',active),('inactive',~active),('all',torch.ones_like(active))]:
                sat[name+'_'+suffix+'_'+group+'_count']=int((mask&selection).sum())
    row=dict(label=LABEL,index=i,low=r['low'],winner=state['winner'],best_step=state['best_step'],
        identity_best_step=best_steps[0],selected_best_step=best_steps[1],identity_updates=500,selected_updates=500,
        active_count=int(active.sum()),reproduction_max_abs=pre['rows'][i]['reproduction_max_abs'],grid_max_abs=pre['rows'][i]['grid_max_abs'],
        seconds=binding['seconds'],**sat)
    for name,s in scores.items():row.update({name+'_'+k:v for k,v in s.items()})
    row.update(delta_psnr=scores['oracle']['psnr']-scores['selected']['psnr'],delta_ssim=scores['oracle']['ssim']-scores['selected']['ssim']);rows.append(row)
with (a.out/'per_image.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
write(a.out/'per_image.json',dict(label=LABEL,rows=rows))
assert max(errors)<1e-11 and len(opened)==200
write(a.out/'evaluation_receipt.json',dict(label=LABEL,completed_utc=utc(),all100_frozen_before_scoring=True,
    freeze_sha256=sha(a.out/'freeze.json'),all200_histories_finite_and_in_bounds=True,
    all_winners_earliest_MSE=True,independent_metric_max_abs_error=max(errors),opened_validation_images=opened,
    all_accepted_metrics_reproduced=True,official_test_opened=False))
print('Postfreeze scoring and independent metric checks PASS',flush=True)
