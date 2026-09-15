"""No labels, metrics, reference gradients or images are inputs to this process."""
from core import *
import argparse,sys,time
p=argparse.ArgumentParser()
for k in ['accepted','manifest','binding','out']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();torch.set_num_threads(1);start=time.perf_counter()
binding=json.loads(a.binding.read_bytes())
for name,h in binding.items():assert sha(name)==h,name
opened=[]
def record(event,args):
    if event!='open' or not isinstance(args[0],str):return
    q=Path(args[0]).resolve()
    if q.suffix.lower() in ['.png','.jpg','.jpeg']:raise AssertionError('No image opens in T044')
    if q.is_relative_to(a.accepted.resolve()):
        assert q.name in ['freeze.json','config.json','decision.json','trajectory.pt'],str(q)
        opened.append(str(q))
sys.addaudithook(record)
assert sha(a.manifest)==COHORT and sha(a.accepted/'freeze.json')==FREEZE
f=json.loads((a.accepted/'freeze.json').read_bytes());m=json.loads(a.manifest.read_bytes())['selected']
assert f['manifest_sha256']==COHORT and sha(a.accepted/'config.json')==f['config_sha256']
a.out.mkdir(parents=True,exist_ok=False);rows=[];snapshots=[]
assert len(m)==len(f['rows'])==100
for i,(item,old) in enumerate(zip(m,f['rows'])):
    assert old['index']==i and old['low']==item['low']
    root=a.accepted/f'{i:03d}'/'common';files=old['methods']['common']['files']
    for name in ['decision.json','trajectory.pt']:assert sha(root/name)==files[name]['sha256']
    d=json.loads((root/'decision.json').read_bytes());t=torch.load(root/'trajectory.pt',map_location='cpu',weights_only=True)
    selected=d['selection']['selected_step'];assert selected==old['methods']['common']['selected_step']
    assert t['states'].shape==(41,1,3,2,2)
    anchor=t['states'][10].clone();state=t['states'][selected].clone();active=d['gate']['active']
    assert torch.isfinite(anchor).all() and torch.isfinite(state).all()
    gate_sha=hashlib.sha256(json.dumps(d['gate'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    value=score(anchor,state,active)
    rows.append(dict(index=i,low=item['low'],selected_step=selected,anchor_step=10,active=active,active_regions=sum(active),zero_active=not any(active),D_legacy=value,gate_sha256=gate_sha,anchor_sha256=thash(anchor),selected_sha256=thash(state),anchor_legacy_sha256=thash(anchor[:,:2]),selected_legacy_sha256=thash(state[:,:2]),trajectory_sha256=files['trajectory.pt']['sha256'],decision_sha256=files['decision.json']['sha256']))
    snapshots.append(dict(anchor=anchor,selected=state,gate=d['gate']))
torch.save(snapshots,a.out/'states.pt');write(a.out/'scores.json',rows)
write(a.out/'freeze.json',dict(completed_utc=utc(),count=100,definition=DEFINITION,precision='accepted physical_parameters mapping evaluated in float64 from frozen float32 latent tensors',scores_sha256=sha(a.out/'scores.json'),states_sha256=sha(a.out/'states.pt'),cohort_sha256=COHORT,prior_freeze_sha256=FREEZE,source_binding=binding,source_binding_sha256=sha(a.binding),accepted_root=str(a.accepted),opened_accepted_files=opened,normal_opens=0,prior_metric_opens=0,prior_reference_gradient_opens=0,loss_id_opens=0,optimizer_updates=0,selection_changes=0,official_test_access=False,zero_active_indices=[r['index'] for r in rows if r['zero_active']],seconds=time.perf_counter()-start))
print('SCORES FROZEN',sha(a.out/'freeze.json'),flush=True)
