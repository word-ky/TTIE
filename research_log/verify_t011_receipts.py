"""Recompute offline T011 reports from transferred records; no model execution."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ttie.projected_metrics import summarize
from t011_projection_diagnostics import diagnose

root=Path(__file__).resolve().parent/'remote_runs/20260912-135142-ttie-t011-a6000/artifacts/audit'
artifacts=json.loads((root/'artifact_manifest.json').read_text())
rows=json.loads((root/'metrics.json').read_text());assert len(artifacts)==240 and len(rows)==2160
config=json.loads((root/'config.json').read_text());count=0
for entry in artifacts:
    directory=root/entry['directory']
    for name in ('states','decisions'):
        receipt=entry[name];path=directory/receipt['file']
        assert path.stat().st_size==receipt['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==receipt['sha256']
        count+=1
    case_rows=json.loads((directory/'metrics.json').read_text())
    assert len(case_rows)==9 and [r['method'] for r in case_rows]==config['methods']
    assert case_rows==[r for r in rows if r['image_id']==entry['image_id'] and r['condition']==entry['condition']]
differences=[]
def compare(a,b,path='root'):
    if isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for key in a:compare(a[key],b[key],path+'.'+key)
    elif isinstance(a,list):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+f'[{i}]')
    elif isinstance(a,float):
        difference=abs(a-b)
        if difference:differences.append(dict(path=path,absolute_difference=difference))
        assert difference<=1e-12,path
    else:assert a==b,path
summary=json.loads((root/'summary.json').read_text())
compare(summary,summarize(rows),'summary')
compare(json.loads((root/'projection_diagnostics.json').read_text()),diagnose(root),'projection_diagnostics')
report=dict(task='T011',archive_sha256='88a63488c7e03010d85f764066bc39bf31fad300489880776a2102ffef674fee',
    verified_local_state_decision_files=count,exact_case_row_membership=True,summary_and_diagnostics_recomputed=True,
    numeric_absolute_tolerance=1e-12,nonidentical_float_reductions=differences,
    max_absolute_difference=max((r['absolute_difference'] for r in differences),default=0.),
    qualified=summary['qualified'],failed=summary['failed'],model_or_output_regeneration=False)
(root/'receipt_verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
