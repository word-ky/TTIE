"""Separate process: references can first open only after all80 actions are durable."""
import argparse,math,torch
from ttie.energy_model import load_energy
from ttie.sobolev_train import cosine
from ttie.natural import load_image
from research_log.T059B.fit import detail_statistics
from research_log.T059E.core import selected_rows
from research_log.T059S.core import *
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out;torch.set_num_threads(1)
freeze=json.loads((out/'action_freeze.json').read_bytes());assert freeze['actions']==80
for n,h in freeze['files'].items():assert sha(out/n)==h
for n,h in freeze['input_hashes_before'].items():assert sha(n)==h
for n,h in freeze['source_bindings'].items():assert sha(n)==h
first=utc();atomic_json(out/'reference_open.json',dict(first_reference_read_utc=first,action_freeze_utc=freeze['utc'],action_freeze_sha256=sha(out/'action_freeze.json')));assert first>freeze['utc']
actions=json.loads((out/'actions.json').read_bytes());field=torch.load(out/'field.pt',weights_only=True,map_location='cpu');wanted={r['index'] for r in actions}
archive=json.loads(Path('research_log/T058AF_archives.json').read_bytes());refs={};reads=[]
for n,h in sorted(archive['files'].items()):
    if n.startswith('artifacts/T058AF/reference_') and n.endswith('.pt'):
        subset,receipt=selected_rows(RR/n,wanted,'g_R',(1,1,8,8));refs.update(subset)
        if receipt['ranges']:reads.append(dict(**receipt,accepted_full_chunk_sha256=h))
assert set(refs)==wanted
assert sha(E/'heldout_access.json')=='75e4ce2b3ca3da31e179caa40293c2b3c0d62546099673ddfd9340412f537731'
access=json.loads((E/'heldout_access.json').read_bytes());accepted={r['index']:r['sha256'] for c in access['cache_reads'] if c['key']=='g_R' for r in c['ranges']}
for i,g in refs.items():assert thash(g)==accepted[i]
assert sha(E/'heldout_statistics.json')=='3673efcfee8872947c53450afc7af948f50d5ad51708768dc3ef2d8150a0122c'
old=json.loads((E/'heldout_statistics.json').read_bytes())['detail'];assert thash(field['g'])==old['gradient_sha256']
assert thash(field['x'])==access['tensor_hashes']['x'] and thash(field['J'])==access['tensor_hashes']['detail_jacobian']
truth=torch.zeros_like(field['g']);mask=torch.zeros(len(truth),dtype=torch.bool)
for r in actions:truth[r['position']]=refs[r['index']].flatten();mask[r['position']]=refs[r['index']].double().norm()>1e-12
head=load_energy(E/'head.pt').eval().requires_grad_(False)
replay=detail_statistics(head,field['x'],field['J'],truth,mask)
cos=cosine(field['g'][mask],truth[mask]);direct=dict(positive_fraction=float((cos>0).float().mean()),median_cosine=float(torch.quantile(cos,.5)))
assert all(replay[k]==v for k,v in direct.items()) and replay['gradient_sha256']==old['gradient_sha256']
atomic_json(out/'gradient_replay.json',dict(anchor=replay,direct=direct,full_accepted_E_gradient_hash_matches=True,reference_ranges=reads,eligible_positions=mask.nonzero().flatten().tolist(),accepted_all1529_detail_statistic=old,scope='Anchor-only references; original E function applied to original1529 feature/J batch with anchor-only eligibility mask'))
manifest=Path('research_log/T014_source_manifest.json');assert sha(manifest)=='4dbf8c604bc44574a5823ec8e22142c8ac39574b6103c4bbd0a69647f15a2257'
sources={r['image_id']:r for r in json.loads(manifest.read_bytes())['images'] if r['split']=='train_t014_sobolev'}
cleans={};opens=[];rows=[]
for r in actions:
    image=r['image_id']
    if image not in cleans:
        spec=sources[image];path=ROOT/'shared/t008/val2017'/spec['filename'];stamp=utc();assert sha(path)==spec['sha256'];cleans[image]=load_image(path);opens.append(dict(image_id=image,path=str(path),sha256=spec['sha256'],utc=stamp));atomic_json(out/'clean_opens.json',opens)
    t=torch.load(out/r['file'],weights_only=True,map_location='cpu');clean=cleans[image];assert clean.shape==t['y0'].shape
    m0=float((t['y0'].double()-clean.double()).square().mean());m1=float((t['y1'].double()-clean.double()).square().mean());change=m1-m0
    if m0==0:assert m1==0 and not mask[r['position']]
    rel=change/m0 if m0 else 0.
    rows.append(dict(index=r['index'],bank_index=r['bank_index'],state_index=0,image_id=image,eligible=bool(mask[r['position']]),mse0=m0,mse1=m1,mse_change=change,relative_mse_change=rel,psnr_change=10*math.log10(max(m0,1e-12)/max(m1,1e-12)),reference_gradient_norm=float(refs[r['index']].double().norm()),active_regions=r['active_regions']))
assert len(cleans)==16 and len(rows)==80
save_tensor(out/'reference_gradients.pt',refs);atomic_json(out/'evaluation_table.json',rows)
for n,h in freeze['files'].items():assert sha(out/n)==h
for n,h in freeze['input_hashes_before'].items():assert sha(n)==h
for r in opens:assert sha(r['path'])==r['sha256']
result=dict(**stats(rows),anchor_gradient=direct,action_freeze_utc=freeze['utc'],first_reference_read_utc=first,counters=dict(training_runs=0,new_head_optimizer_steps=0,detail_action_steps=80,clean_or_reference_reads_before_action_freeze=0,outer_supervision_reads=0,target_domain_access=0,lolv2_access=0,official_test_access=0,inference_reference_leakage=0),immutable_inputs=True,completed_utc=utc())
atomic_json(out/'result.json',result);print(json.dumps(result),flush=True)
