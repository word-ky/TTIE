"""Independent SciPy correlation audit, separate from the Torch runtime."""
import json
from pathlib import Path
import statistics as st
import subprocess
from scipy.stats import spearmanr

root=Path('research_log/T016D_run')
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
item=read(root/'config.json')['input_artifact_hashes']['T016B/reference']
refs={r['episode']:r for r in json.loads(subprocess.check_output(['git','show',item['commit']+':'+item['path']]))}
summary=read(root/'summary.json'); counts={}
for name,rows in read(root/'evaluation.json').items():
    defined=0
    for row in rows:
        values=row['predictions'];mse=refs[row['episode']]['reference_mse']
        if len(set(values))>1 and len(set(mse))>1:
            expected=float(spearmanr(values,mse).statistic)
            assert abs(expected-row['spearman'])<1e-12; defined+=1
        else:assert row['spearman'] is None
    counts[name]=dict(defined=defined,null=len(rows)-defined)
    for condition,g in summary['probes'][name]['groups'].items():
        group=rows if condition=='spatial_pool' else [r for r in rows if r['condition']==condition]
        valid=[r['spearman'] for r in group if r['spearman'] is not None]
        expected=dict(count=len(group),null_count=len(group)-len(valid),mean=st.mean(valid),median=st.median(valid),min=min(valid),max=max(valid))
        assert g['spearman']==expected
result=dict(passed=True,scipy_correlations=counts,no_torch_import=True,no_training_or_prediction_changes=True)
Path('research_log/T016D_correlation_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))
