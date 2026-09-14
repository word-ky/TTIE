"""Independent NumPy reduction from saved full gradients; no image/model execution."""
import json
from pathlib import Path
import argparse
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--preflight',type=Path,required=True)
a=p.parse_args();rows=json.loads((a.out/'states.json').read_bytes());summary=json.loads((a.out/'summary.json').read_bytes())
pre=json.loads((a.preflight/'preflight.json').read_bytes());g=torch.load(a.out/'gradients.pt',weights_only=True,map_location='cpu').numpy().astype(np.float64)
rebuilt=[];max_error=0.
for row in rows:
    i,s=row['index'],row['step'];mask=np.tile(pre['rows'][i]['gate']['active'],2).astype(bool)
    e=g[i,s,0].reshape(-1)[mask];r=g[i,s,1].reshape(-1)[mask]
    en=np.sqrt(np.sum(e*e));rn=np.sqrt(np.sum(r*r));dot=np.sum(e*r);deg=en<=1e-12 or rn<=1e-12
    cosine=None if deg else dot/en/rn
    for key,v in [('energy_norm',en),('reference_norm',rn),('dot',dot),('cosine',cosine)]:
        if v is not None:
            error=abs(row[key]-v);max_error=max(max_error,error);assert np.isclose(row[key],v,rtol=1e-12,atol=1e-12)
    assert row['degenerate']==deg and row['positive_dot']==bool(dot>0)
    rebuilt.append(dict(**row,check_cosine=cosine))
groups=[('all_states',rebuilt),('frozen_selected',[r for r in rebuilt if r['frozen_selected']])]
groups += [(str(s),[r for r in rebuilt if r['step']==s]) for s in range(41)]
for name,rs in groups:
    target=summary[name] if name in ['all_states','frozen_selected'] else summary['by_step'][name]
    valid=[r for r in rs if r['check_cosine'] is not None];c=np.array([r['check_cosine'] for r in valid])
    values=dict(count=len(rs),nondegenerate=len(valid),cosine_mean=float(np.mean(c)),cosine_median=float(np.median(c)),
        cosine_p10=float(np.percentile(c,10)),cosine_p90=float(np.percentile(c,90)),positive_dot_fraction=np.mean([r['dot']>0 for r in valid]),
        energy_zero_fraction=np.mean([r['energy_norm']<=1e-12 for r in rs]),reference_zero_fraction=np.mean([r['reference_norm']<=1e-12 for r in rs]),
        either_zero_fraction=np.mean([r['degenerate'] for r in rs]))
    for key,v in values.items():assert np.isclose(v,target[key],rtol=1e-12,atol=1e-12)
samples=json.loads((a.out/'independent_samples.json').read_bytes())
sample_errors={key:0. for key in ['energy_norm','reference_norm','dot','cosine']}
for sample in samples:
    row=rows[sample['index']*41+sample['step']]
    for key in ['energy_norm','reference_norm','dot','cosine']:
        sample_errors[key]=max(sample_errors[key],abs(sample[key]-row[key]))
        # These use independently recomputed float32 CUDA gradients, unlike the
        # float64 NumPy reductions above (which remain checked at1e-12).
        assert np.isclose(sample[key],row[key],rtol=1e-4,atol=1e-6)
result=dict(status='PASS',states=len(rows),summary_groups=len(groups),independent_gradient_samples=len(samples),max_scalar_abs_error=max_error,
    sample_max_abs_errors=sample_errors,sample_rtol=1e-4,sample_atol=1e-6,aggregate_rtol=1e-12,aggregate_atol=1e-12)
(a.out/'independent_check.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
