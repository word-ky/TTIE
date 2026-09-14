"""REFERENCE_ORACLE_ONLY:200fixed starts, then freeze before quality metrics."""
from core import *
from core import prior as o
argparse=o.argparse;json=o.json;time=o.time;sha=o.sha;write=o.write;utc=o.utc;native=o.native;LABEL=o.LABEL;SPLIT_SHA=o.SPLIT_SHA;Image=o.Image

p=argparse.ArgumentParser(description=LABEL)
for n in ['split','low-root','normal-root','accepted','preflight','out']:p.add_argument('--'+n,type=Path,required=True)
a=p.parse_args();assert sha(a.split)==SPLIT_SHA
split=json.loads(a.split.read_bytes())['selected'];pre=json.loads((a.preflight/'preflight.json').read_bytes())
assert pre['all100_identity_wb_regression'] and len(pre['rows'])==100
assert sha(a.preflight/'bound_starts.pt')==pre['starts_sha256']
starts=torch.load(a.preflight/'bound_starts.pt',weights_only=True)['starts']
# This isolated oracle permits existing validation references only after WB regression.
assert pre['normal_decodes']==0 and pre['renderer_max_abs']<=1e-6
assert sha(a.accepted/'freeze.json')==pre['accepted_freeze_sha256']
allowed={str((root/r[k]).resolve()) for r in split for root,k in [(a.low_root,'low'),(a.normal_root,'normal')]}
opened=[];original=Image.open
def validation_only(path,*args,**kw):
    resolved=str(Path(path).resolve());assert resolved in allowed
    opened.append(resolved);return original(path,*args,**kw)
Image.open=validation_only
torch.manual_seed(7);torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
assert torch.cuda.is_available();a.out.mkdir(parents=True,exist_ok=False)
config=dict(label=LABEL,task='T034-A',seed=7,optimizer='Adam',lr=.05,updates_per_start=500,
    starts=['identity','T026A_selected'],selection='minimum reference MSE, earliest step, identity on remaining tie',
    objective='full-frame RGB MSE float64 accumulation through float32 renderer',
    bounds='exact T026A EVgamma plus activeRGBWB.5..2 inactiveidentity; contrast1',
    created_utc=utc(),gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
    preflight_sha256=sha(a.preflight/'preflight.json'),
    split_sha256=sha(a.split),code_sha256={str(p):sha(p) for p in Path('research_log/T034A_oracle').glob('*.py')})
write(a.out/'config.json',config);rows=[]
for i,r in enumerate(split):
    start=time.perf_counter();d=a.accepted/f'{i:03d}';binding=pre['rows'][i]
    assert r['low']==binding['low']
    assert all(sha(d/n)==h for n,h in binding['files'].items())
    assert sha(a.low_root/r['low'])==r['low_sha256'] and sha(a.normal_root/r['normal'])==r['normal_sha256']
    low=native(a.low_root/r['low']).cuda()
    decision=json.loads((d/'decision.json').read_bytes());model,box=frozen_model(decision,low)
    # All100 exact reconstructions and two starts were frozen in the prior process.
    reference=native(a.normal_root/r['normal']).cuda()
    results=[];start_outputs=[]
    for initial in starts[i]:
        result=optimize_start(model,box,low,reference,initial.cuda());results.append(result)
        with torch.no_grad():start_outputs.append(model(low).cpu())
    winner=min(range(2),key=lambda j:(results[j]['best_mse'],results[j]['best_step'],j));best=results[winner]
    with torch.no_grad():
        model.raw.copy_(best['best_raw'].cuda());output=model(low);grid=model.physical_grid()[:,:5]
    assert best['best_mse']<=results[1]['initial_mse']+1e-10
    dest=a.out/f'{i:03d}';dest.mkdir()
    torch.save(dict(label=LABEL,image=output.cpu(),raw=best['best_raw'],grid=grid.cpu()),dest/'oracle_output.pt')
    torch.save(dict(label=LABEL,images=torch.stack(start_outputs)),dest/'start_outputs.pt')
    torch.save(dict(label=LABEL,starts=[dict(raw=x['raw_history'],mse=torch.tensor(x['history'],dtype=torch.float64)) for x in results]),dest/'histories.pt')
    states=dict(label=LABEL,index=i,low=r['low'],gate=decision['gate'],box=decision['diagnostics']['action_box'],
        winner=['identity','T026A_selected'][winner],best_step=best['best_step'],raw=best['best_raw'].tolist(),physical_grid=grid.cpu().tolist(),
        starts=[{k:(v.tolist() if isinstance(v,torch.Tensor) else v) for k,v in x.items() if k not in ['raw_history','history']} for x in results])
    write(dest/'oracle_states.json',states)
    rows.append(dict(index=i,low=r['low'],seconds=time.perf_counter()-start,winner=states['winner'],best_step=best['best_step'],
        files={n:sha(dest/n) for n in ['oracle_output.pt','oracle_states.json','histories.pt','start_outputs.pt']}))
    write(a.out/'progress.json',dict(label=LABEL,completed=i+1,utc=utc(),last=rows[-1]))
    print(f'{LABEL} {i+1}/100 winner={states["winner"]} step={best["best_step"]} seconds={rows[-1]["seconds"]:.2f}',flush=True)
assert len(opened)==200 and set(opened)==allowed
assert sha(a.accepted/'freeze.json')==pre['accepted_freeze_sha256']
write(a.out/'freeze.json',dict(label=LABEL,utc=utc(),pairs=100,starts=200,updates=100000,
    states_per_start=501,all_states_finite_and_in_bounds=True,opened_validation_images=opened,official_test_opened=False,
    rows=rows,config_sha256=sha(a.out/'config.json')))
print('FROZEN100oracleoutputs/200histories; PSNR/SSIM not yet computed',flush=True)
