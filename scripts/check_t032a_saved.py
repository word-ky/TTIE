"""NumPy/SciPy-only independent compact evidence replay; no Torch runtime."""
import argparse,json,hashlib,csv,math
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.spatial.distance import cdist
p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--arrays',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
audit=a.result/'audit';radius=a.result/'radius';f=json.loads((audit/'freeze.json').read_bytes());rf=json.loads((radius/'freeze.json').read_bytes());spec=json.loads((radius/'spec.json').read_bytes())
assert sha(radius/'freeze.json')==spec['radius_freeze_sha256']==f['radius_freeze_sha256']
assert sha(radius/'spec.json')==f['spec_sha256'] and spec['r_support']==rf['r_support']
assert sha(radius/'source.pt')==rf['source_sha256'] and sha(radius/'cross_distances.json')==rf['cross_distances_sha256']
deployment=json.loads((a.result/'preparation/reference_deployment.json').read_bytes())
assert sha(audit/'freeze.json')==deployment['freeze_sha256'] and f['completed_utc']<deployment['started_utc']
b=np.load(a.arrays);z=(b['source_features'].astype('float64')-b['x_mean'].astype('float64'))/b['x_scale'].astype('float64')
q=(b['features'].astype('float64')-b['x_mean'].astype('float64'))/b['x_scale'].astype('float64');offset=0;error=0.;decisions=[]
for row,n in zip(f['rows'],b['lengths']):
    d=audit/f'{row["index"]:03d}'
    for name in ['decision.json','trajectory.pt']:assert sha(d/name)==row['files'][name]['sha256']
    distances=cdist(q[offset:offset+n],z).min(1)/math.sqrt(28)
    error=max(error,float(np.abs(distances-b['distances'][offset:offset+n]).max()));offset+=n
    r=json.loads((d/'decision.json').read_bytes());energies=np.asarray(r['original']['scores']);original=int(np.argmin(energies))
    exits=np.flatnonzero(distances>rf['r_support']);exit_step=int(exits[0]) if len(exits) else None
    cutoff=len(distances)-1 if exit_step is None else max(0,exit_step-1);chosen=int(np.argmin(energies[:cutoff+1]))
    expected=dict(original_step=original,selected_step=chosen,cutoff=cutoff,exit_step=exit_step)
    assert expected==r['guarded'] and original==r['original']['selected_step']
    assert row['original_distance']==b['distances'][offset-n+original] and row['selected_distance']==b['distances'][offset-n+chosen]
    decisions.append(expected)
assert error<=1e-9
rows=list(csv.DictReader((audit/'metrics.csv').open()));s=json.loads((audit/'summary.json').read_bytes());aggregation_error=0.
for key in ['original_psnr','original_ssim','guarded_psnr','guarded_ssim','delta_psnr','delta_ssim']:
    values=np.asarray([float(r[key]) for r in rows]);actual={'mean':np.mean(values),'median':np.median(values),'p10':np.quantile(values,.1),'p90':np.quantile(values,.9)}
    aggregation_error=max(aggregation_error,max(abs(actual[k]-s[key][k]) for k in actual))
for m in ['psnr','ssim']:
    delta=np.asarray([float(r['guarded_'+m])-float(r['original_'+m]) for r in rows]);assert np.array_equal(delta,[float(r['delta_'+m]) for r in rows])
    assert s['win_equal_loss'][m]==dict(win=int((delta>0).sum()),equal=int((delta==0).sum()),loss=int((delta<0).sum()))
assert s['changed_selection_count']==sum(d['original_step']!=d['selected_step'] for d in decisions)
assert s['first_exit_histogram']==dict(Counter(str(d['exit_step']) for d in decisions))
assert s['classification']==('materially positive' if s['delta_psnr']['mean']>=.3 and s['delta_ssim']['mean']>=0 else 'negative/insufficient')
assert aggregation_error<=1e-12
result=dict(status='PASS',states=int(offset),decisions=len(decisions)*2,distance_max_abs_error=error,aggregation_max_abs_error=float(aggregation_error),barrier_verified=True,classification=s['classification'])
a.out.write_text(json.dumps(result,indent=2)+'\n');print(result)
