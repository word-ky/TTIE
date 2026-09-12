"""Independent SciPy rank audit in a process without the PyTorch OpenMP runtime."""
import json
from pathlib import Path
import subprocess
from scipy.stats import spearmanr

root=Path('research_log/T016C_run')
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
config=read(root/'config.json');item=config['input_artifact_hashes']['reference']
reference={r['episode']:r for r in json.loads(subprocess.check_output(['git','show',config['evidence_commit']+':'+item['path']]))}
evaluated=read(root/'evaluation.json');counts={}
for name,rows in evaluated.items():
    oof={r['episode']:r for r in read(root/name/'oof.json')};defined=0
    for row in rows:
        predictions=oof[row['episode']]['predictions'];mse=reference[row['episode']]['reference_mse']
        if len(set(predictions))>1 and len(set(mse))>1:
            expected=float(spearmanr(predictions,mse).statistic)
            assert abs(expected-row['spearman'])<1e-12;defined+=1
        else:assert row['spearman'] is None
    counts[name]=dict(defined=defined,null=len(rows)-defined)
result=dict(passed=True,scipy_correlations=counts,no_torch_import=True,no_training_or_prediction_changes=True)
Path('research_log/T016C_correlation_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))
