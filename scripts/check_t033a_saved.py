"""Independent scalar aggregation and cohort/freeze replay from compact evidence."""
import argparse,json,csv,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser()
for k in ['result','split','ours','out']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
audit=a.result/'audit';f=json.loads((audit/'freeze.json').read_bytes());s=json.loads((audit/'summary.json').read_bytes());ev=json.loads((audit/'evaluation_receipt.json').read_bytes());dep=json.loads((a.result/'evaluation_deployment.json').read_bytes())
assert sha(audit/'freeze.json')==dep['freeze_sha256']==ev['freeze_sha256']
assert sha(a.split)==f['split_sha256'] and sha(a.ours)==ev['ours_csv_sha256']
assert f['completed_utc']<dep['created_utc']<min(r['utc'] for r in ev['normal_opens']) and f['normal_decodes']==0
split=json.loads(a.split.read_bytes())['selected'];ours=list(csv.DictReader(a.ours.open()));rows=list(csv.DictReader((audit/'metrics.csv').open()))
assert len(split)==len(ours)==len(rows)==len(f['rows'])==len(f['opened'])==len(ev['normal_opens'])==100
for i,(r,o,t,h) in enumerate(zip(rows,ours,split,f['rows'])):
    assert r['low']==o['low']==t['low']==h['low'] and int(r['index'])==i and h['shape']==[400,600,3]
    for m in ['psnr','ssim']:
        assert float(r['ours_'+m])==float(o['ours_'+m])
        assert float(r['delta_'+m])==float(r['retinex_'+m])-float(o['ours_'+m])
error=0.
for k,summary in s['metrics'].items():
    v=np.asarray([float(r[k]) for r in rows]);actual=dict(mean=float(np.mean(v)),median=float(np.median(v)),p95=float(np.quantile(v,.95)))
    error=max(error,max(abs(actual[n]-summary[n]) for n in actual))
for m in ['psnr','ssim']:
    v=np.asarray([float(r['delta_'+m]) for r in rows]);assert s['win_equal_loss'][m]==dict(win=int((v>0).sum()),equal=int((v==0).sum()),loss=int((v<0).sum()))
v=np.asarray([float(r['seconds']) for r in rows]);actual=dict(mean=float(np.mean(v)),median=float(np.median(v)),p95=float(np.quantile(v,.95)))
error=max(error,max(abs(actual[n]-s['runtime'][n]) for n in actual));assert error<=1e-12
result=dict(status='PASS',pairs=100,paired_deltas_exact=True,aggregation_max_abs_error=error,cohort_and_freeze_order_verified=True,classification=s['classification'])
a.out.write_text(json.dumps(result,indent=2)+'\n');print(result)
