"""NumPy/SciPy-only distance and validity replay; avoids mixed OpenMP runtimes."""
import argparse,json,math
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr,rankdata
p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--arrays',type=Path,required=True);a=p.parse_args()
b=np.load(a.arrays);f=json.loads((a.result/'support/freeze.json').read_bytes());rows=json.loads((a.result/'support/scores.json').read_bytes())
reference=a.result/'REFERENCE_GRADIENT_DIAGNOSTIC_ONLY';labels=json.loads((reference/'validity.json').read_bytes());summary=json.loads((reference/'summary.json').read_bytes())
mean=b['x_mean'].astype(float);scale=b['x_scale'].astype(float);x=(b['source_features'].astype(float)-mean)/scale;q=(b['state_features'].astype(float)-mean)/scale
d=np.array([r['distance'] for r in rows]);replayed=np.concatenate([cdist(c,x).min(1)/math.sqrt(28) for c in np.array_split(q,33)])
error=float(np.max(np.abs(d-replayed)));assert error<=1e-9
g=b['gradients'].astype(float);cosines=[];valid=[]
for r in labels:
    i,s=r['index'],r['step'];mask=np.tile(f['accepted_bindings'][i]['gate']['active'],2).astype(bool)
    e=g[i,s,0].ravel()[mask];truth=g[i,s,1].ravel()[mask];dot=e@truth;cosine=dot/np.linalg.norm(e)/np.linalg.norm(truth)
    assert bool(dot>0)==r['valid'] and abs(cosine-r['cosine'])<1e-12
    cosines.append(r['cosine']);valid.append(r['valid'])
invalid=~np.array(valid);n=invalid.sum();m=len(d)-n;auc=(rankdata(d)[invalid].sum()-n*(n+1)/2)/(n*m)
rho=spearmanr(d,cosines).statistic
assert auc==summary['auc_invalid'] and rho==summary['spearman_rho']
print(dict(status='PASS',rows=len(d),distance_max_abs_error=error,auc=float(auc),rho=float(rho)))
