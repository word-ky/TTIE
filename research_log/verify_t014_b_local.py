"""Offline verification of fetched T014 fresh receipts; no training/calibration gate."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()))
from ttie.stop_receipt import sha
from ttie.sobolev_metrics import stage_b,summarize_trajectories

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
    case=read(d/'metrics.json');assert len(case)==10 and all(r['image_id']==e['image_id'] and r['condition']==e['condition'] for r in case)
    rows.extend(case);assert not (d/'alignments.json').exists()
assert len(entries)==240 and rows==read(root/'metrics.json') and read(root/'alignments.json')==[]
receipt=read(Path('research_log/T014_energy_receipt.json'));final=read(root/'final_checks.json')
assert sha(Path('research_log/T014_energy.pt'))==receipt['energy_sha256']==final['energy_sha256']['sobolev_primary']
assert sha(Path('research_log/T014_value_only.pt'))==receipt['value_only_sha256']==final['energy_sha256']['value_only_control']
computed=stage_b(rows);computed['trajectory_distributions']=summarize_trajectories(entries);computed=json.loads(json.dumps(computed))
differences=[]
def compare(x,y):
    if isinstance(x,dict):
        assert x.keys()==y.keys()
        for k in x:compare(x[k],y[k])
    elif isinstance(x,list):
        assert len(x)==len(y)
        for i,j in zip(x,y):compare(i,j)
    elif isinstance(x,float):
        delta=abs(x-y);assert delta<=1e-12;differences.append(delta)
    else:assert x==y
report=read(root/'summary.json');compare(computed,report)
result=dict(local_small_hashes_verified=checked,evaluation_rows=len(rows),summary_max_abs_difference=max(differences),
    both_frozen_heads_match=True,qualified=report['qualified'],failed=report['failed'],large_image_bytes_preserved_remotely=large,
    archive_sha256=sha(a.archive),stage='B')
(root/'local_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
