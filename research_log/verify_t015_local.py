"""Verify fetched T015 small receipts against the frozen final archive."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()))
from ttie.routing.metrics import summarize,routing_diagnostics,summarize_trajectories
from ttie.routing.core import BASES,route
from ttie.stop_receipt import sha

p=argparse.ArgumentParser();p.add_argument('audit',type=Path);p.add_argument('--archive',type=Path,required=True);a=p.parse_args()
read=lambda p:json.loads(p.read_text());root=a.audit;entries=read(root/'artifact_manifest.json');rows=[];checked=0;large=0
for e in entries:
    d=root/e['directory'];assert e['files']==read(d/'label_free_receipt.json')
    for group,files in e['files'].items():
        folder=d if group=='episode' else d/group
        for name,r in files.items():
            path=folder/name
            if name in ('checkpoint_images.pt','outputs.pt'):assert not path.exists();large+=r['bytes']
            else:assert path.stat().st_size==r['bytes'] and sha(path)==r['sha256'];checked+=1
    r=route(*(e['selections'][m]['scores'][e['selections'][m]['selected_step']] for m in BASES))
    assert r==e['routing']==read(d/'routing.json')
    case=read(d/'metrics.json');assert len(case)==10;rows.extend(case)
assert len(entries)==240 and rows==read(root/'metrics.json')
config=read(root/'config.json');assert sha(Path('research_log/T014_energy.pt'))==config['frozen_energy_sha256']
assert sha(Path('research_log/T015_manifest.json'))==config['manifest_sha256']
computed=summarize(rows);computed['trajectory_distributions']=summarize_trajectories(entries)
computed=json.loads(json.dumps(computed));differences=[]
def compare(x,y):
    if isinstance(x,dict):
        assert x.keys()==y.keys()
        for k in x:compare(x[k],y[k])
    elif isinstance(x,list):
        assert len(x)==len(y)
        for i,j in zip(x,y):compare(i,j)
    elif isinstance(x,float):
        d=abs(x-y);assert d<=1e-12;differences.append(d)
    else:assert x==y
report=read(root/'summary.json');compare(computed,report)
compare(routing_diagnostics(entries),read(root/'routing_diagnostics.json'))
result=dict(task='T015',local_small_hashes_verified=checked,evaluation_rows=len(rows),
    summary_max_abs_difference=max(differences),frozen_head_manifest_match=True,
    all_routes_margin_counts_regret_recomputed=True,qualified=report['qualified'],failed=report['failed'],
    large_image_bytes_preserved_remotely=large,archive_sha256=sha(a.archive))
(root/'local_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
