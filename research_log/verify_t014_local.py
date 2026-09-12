"""Verify downloaded T014 small receipts and reproduce calibration conclusions."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()))
from ttie.stop_receipt import sha
from ttie.sobolev_metrics import stage_a,summarize_trajectories

p=argparse.ArgumentParser();p.add_argument('audit',type=Path);p.add_argument('--archive',type=Path,required=True);args=p.parse_args()
root=args.audit;read=lambda p:json.loads(p.read_text());checked=0;large_bytes=0
omitted={'bank_images.pt','checkpoint_images.pt','outputs.pt'}
def check(directory,files):
    global checked,large_bytes
    for name,r in files.items():
        file=directory/name
        if name in omitted:
            assert not file.exists();large_bytes+=r['bytes']
        else:
            assert file.stat().st_size==r['bytes'] and sha(file)==r['sha256'];checked+=1
for e in read(root/'training_manifest.json'):
    check(root/e['directory'],e['files']);check(root/e['directory'],e['source_supervision_files'])
rows=[];alignments=[];entries=read(root/'artifact_manifest.json')
for e in entries:
    d=root/e['directory'];assert e['files']==read(d/'label_free_receipt.json')
    for group,files in e['files'].items():check(d if group=='episode' else d/group,files)
    case=read(d/'metrics.json');assert len(case)==10 and all(r['image_id']==e['image_id'] and r['condition']==e['condition'] for r in case)
    rows.extend(case)
    if (d/'alignments.json').exists():alignments.extend(read(d/'alignments.json'))
assert rows==read(root/'metrics.json') and alignments==read(root/'alignments.json')
receipt=read(root/'T014_energy_receipt.json')
assert sha(root/'sobolev_primary.pt')==receipt['energy_sha256'] and sha(root/'value_only_control.pt')==receipt['value_only_sha256']
assert sha(root/'training_manifest.json')==receipt['source_records_manifest_sha256']
differences=[]
def compare(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:compare(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):compare(x,y)
    elif isinstance(a,float):
        delta=abs(a-b);assert delta<=1e-12;differences.append(delta)
    else:assert a==b
computed=stage_a(rows,alignments);computed['trajectory_distributions']=summarize_trajectories(entries)
computed=json.loads(json.dumps(computed));report=read(root/'summary.json');compare(computed,report)
result=dict(local_small_hashes_verified=checked,sobolev_sha256=receipt['energy_sha256'],value_only_sha256=receipt['value_only_sha256'],
    calibration_rows=len(rows),alignment_count=len(alignments),summary_max_abs_difference=max(differences),passes=report['passes'],failed=report['failed'],
    large_image_bytes_preserved_remotely=large_bytes,archive_sha256=sha(args.archive),
    no_local_fresh_manifest=not Path('research_log/T014_manifest.json').exists())
(root/'local_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
