import argparse,torch
from ttie.natural import load_image
from research_log.T059U.core import *
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);out=p.parse_args().out;torch.set_num_threads(1)
f=json.loads((out/'action_freeze.json').read_bytes());assert f['actions']==80 and f['tau']==TAU
for n,h in f['files'].items():assert sha(out/n)==h
for n,h in f['input_hashes_before'].items():assert sha(n)==h
stamp=utc();assert stamp>f['utc'];atomic_json(out/'reference_open.json',dict(first_outer_reference_read_utc=stamp,freeze_utc=f['utc'],freeze_sha256=sha(out/'action_freeze.json')))
manifest=Path('research_log/T014_source_manifest.json');assert sha(manifest)=='4dbf8c604bc44574a5823ec8e22142c8ac39574b6103c4bbd0a69647f15a2257'
sources={r['image_id']:r for r in json.loads(manifest.read_bytes())['images'] if r['split']=='train_t014_sobolev'}
actions=json.loads((out/'actions.json').read_bytes());split=json.loads((out/'split.json').read_bytes());cleans={};opens=[];rows=[]
for a in actions:
    i=a['image_id'];assert i in split['image_ids']
    if i not in cleans:
        spec=sources[i];path=ROOT/'shared/t008/val2017'/spec['filename'];assert sha(path)==spec['sha256'];opens.append(dict(image_id=i,path=str(path),sha256=spec['sha256'],utc=utc()));atomic_json(out/'clean_opens.json',opens);cleans[i]=load_image(path)
    t=torch.load(out/a['file'],weights_only=True,map_location='cpu');clean=cleans[i].double();m={k:float((t[k].double()-clean).square().mean()) for k in ['y0','y1','y_gated']};assert all(np.isfinite(x) for x in m.values())
    rows.append(dict(index=a['index'],bank_index=a['bank_index'],state_index=0,image_id=i,norm=a['norm'],acted=a['acted'],tau=TAU,mse0=m['y0'],mse_ungated=m['y1'],mse_gated=m['y_gated'],ungated_A=m['y1']-m['y0'],gated_A=m['y_gated']-m['y0']))
assert len(rows)==80 and set(cleans)==set(split['image_ids'])
result=summary(rows);result.update(counters=dict(training_runs=0,new_heads=0,head_optimizer_steps=0,ungated_comparison_steps=80,gated_acted=sum(r['acted'] for r in rows),premature_outer_reference_reads=0,outer_source_clean_images=16,reference_gradient_reads=0,target_domain_access=0,lolv2_access=0,official_test_access=0,inference_reference_leakage=0),freeze_utc=f['utc'],first_outer_reference_read_utc=stamp,completed_utc=utc())
for n,h in f['files'].items():assert sha(out/n)==h
for n,h in f['input_hashes_before'].items():assert sha(n)==h
for r in opens:assert sha(r['path'])==r['sha256']
atomic_json(out/'evaluation_table.json',rows);atomic_json(out/'result.json',result);print(json.dumps(result),flush=True)
