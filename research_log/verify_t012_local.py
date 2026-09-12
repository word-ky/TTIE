"""Verify the fetched T012 evidence and recompute reference-only summaries."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()))
from ttie.stop_receipt import sha
from ttie.stop_metrics import stage_a,choose_fixed_step

root=Path('research_log/remote_runs/20260912-151505-ttie-t012-stage-a/artifacts/audit')
read=lambda p:json.loads(p.read_text())
entries=read(root/'artifact_manifest.json');rows=[];checked=0;remote_bytes=0
omitted={'checkpoint_images.pt','outputs.pt','selected_outputs.pt'}
for e in entries:
    d=root/e['directory']
    receipts=[read(d/'label_free_receipt.json')]
    if e['split']=='calibration_t012_stop':
        receipts.append(read(d/'selection_receipt.json'))
        case=read(d/'calibration_metrics.json')
        assert len(case)==8 and all(r['image_id']==e['image_id'] and r['condition']==e['condition'] for r in case)
        rows.extend(case)
    for receipt in receipts:
        for name,record in receipt.items():
            if name in omitted:
                assert not (d/name).exists();remote_bytes+=record['bytes'];continue
            assert sha(d/name)==record['sha256'] and (d/name).stat().st_size==record['bytes'];checked+=1
assert rows==read(root/'calibration_metrics.json')
head_receipt=read(root/'T012_stopping_receipt.json')
assert sha(root/'head.pt')==head_receipt['head_sha256']
differences=[]
def compare(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:compare(a[key],b[key])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):compare(x,y)
    elif isinstance(a,float):
        diff=abs(a-b);assert diff<=1e-12;differences.append(diff)
    else:assert a==b
report=read(root/'summary.json');compare(stage_a(rows),report)
fixed=read(root/'fixed_step_calibration.json')
compare(choose_fixed_step(fixed['rows']),{k:v for k,v in fixed.items() if k!='rows'})
g=report['groups']['heterogeneous'];mse=lambda m:g[m]['mse']['mean']
oracle=mse('oracle_best_checkpoint');fixed_mse=mse('fixed_step_source');discrete=mse('region2_discrete_projected')
bound=dict(scope='20 calibration images; 40 heterogeneous episodes; frozen saved T011 checkpoints only',
    oracle_mse=oracle,fixed_step_mse=fixed_mse,discrete_mse=discrete,
    oracle_over_fixed=oracle/fixed_mse,oracle_gain_over_fixed=1-oracle/fixed_mse,
    oracle_over_discrete=oracle/discrete,oracle_gain_over_discrete=1-oracle/discrete,
    perfect_selection_still_fails_fixed_clause=oracle/fixed_mse>.95,
    perfect_selection_still_fails_discrete_clause=oracle/discrete>.95)
result=dict(local_hashes_verified=checked,head_sha256=head_receipt['head_sha256'],
    calibration_rows=len(rows),summary_max_abs_difference=max(differences),passes=report['passes'],
    fixed_step=fixed['selected_step'],large_output_bytes_preserved_remotely=remote_bytes,
    archive_sha256=sha(Path('.autodl/T012_final_receipts.tar.gz')),
    no_local_fresh_manifest=not Path('research_log/T012_manifest.json').exists(),oracle_bound=bound)
(root/'local_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
