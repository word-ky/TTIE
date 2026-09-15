"""All100 exact frozen T035 winner reconstructions, before any normal decode."""
from core import *
import argparse,time
p=argparse.ArgumentParser()
for k in ['split','low-root','accepted','binding','out']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();start=time.perf_counter();torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
b=json.loads(a.binding.read_bytes())
for name,h in b.items():assert sha(name)==h,name
assert sha(a.split)==SPLIT_SHA and sha(a.accepted/'freeze.json')==PRIOR_FREEZE
f=json.loads((a.accepted/'freeze.json').read_bytes());split=json.loads(a.split.read_bytes())['selected']
assert len(split)==len(f['rows'])==100 and sha(a.accepted/'config.json')==f['config_sha256']
config=json.loads((a.accepted/'config.json').read_bytes())
for name,h in config['code_sha256'].items():assert sha(name)==h,name
allowed={str((a.low_root/r['low']).resolve()) for r in split};opened=[];original=o.Image.open
def low_only(path,*args,**kwargs):
    name=str(Path(path).resolve());assert name in allowed;opened.append(name);return original(path,*args,**kwargs)
o.Image.open=low_only;a.out.mkdir(parents=True,exist_ok=False);rows=[];starts=[]
for i,(r,old) in enumerate(zip(split,f['rows'])):
    assert old['index']==i and r['low']==old['low'];root=a.accepted/f'{i:03d}'
    for name,h in old['files'].items():assert sha(root/name)==h,name
    state=json.loads((root/'oracle_states.json').read_bytes());saved=torch.load(root/'oracle_output.pt',weights_only=True,map_location='cpu');hist=torch.load(root/'histories.pt',weights_only=True,map_location='cpu')['starts']
    steps=[int(h['mse'].argmin()) for h in hist];winner=min(range(2),key=lambda j:(float(hist[j]['mse'][steps[j]]),steps[j],j))
    assert state['winner']==['identity','T026A_selected'][winner] and state['best_step']==steps[winner]
    assert torch.equal(saved['raw'],hist[winner]['raw'][steps[winner]]) and torch.equal(saved['raw'],torch.tensor(state['raw']))
    assert sha(a.low_root/r['low'])==r['low_sha256'];low=native(a.low_root/r['low']).cuda()
    decision=dict(gate=state['gate'],diagnostics=dict(action_box=state['box']));model,box=frozen_model(decision,low)
    with torch.no_grad():model.raw.copy_(saved['raw'].cuda());output=model(low).cpu()
    error=float((output-saved['image']).abs().max());assert error<=1e-6
    rows.append(dict(index=i,low=r['low'],gate=state['gate'],box=state['box'],prior_files=old['files'],prior_winner=state['winner'],prior_best_step=state['best_step'],raw_sha256=thash(saved['raw']),accepted_output_sha256=thash(saved['image']),reconstructed_output_sha256=thash(output),max_abs_error=error,bit_exact=torch.equal(output,saved['image'])))
    starts.append(saved['raw'].clone())
torch.save(torch.stack(starts),a.out/'starts.pt');write(a.out/'preflight.json',dict(label=LABEL,completed_utc=utc(),rows=rows,normal_decodes=0,opened_lows=opened,starts_sha256=sha(a.out/'starts.pt'),prior_freeze_sha256=PRIOR_FREEZE,split_sha256=SPLIT_SHA,source_binding=b,source_binding_sha256=sha(a.binding),all100_max_abs=max(r['max_abs_error'] for r in rows),all100_bit_exact=all(r['bit_exact'] for r in rows),seconds=time.perf_counter()-start))
assert len(opened)==100
print('PREFLIGHT100 PASS',sha(a.out/'preflight.json'),flush=True)
