"""Independently recompute paired and grouped summaries from retained scalars."""
import csv,json,hashlib,math,statistics,argparse
from pathlib import Path
from collections import Counter
p=argparse.ArgumentParser();p.add_argument('evidence',type=Path);a=p.parse_args()
rows=list(csv.DictReader((a.evidence/'per_image.csv').open()))
summary=json.loads((a.evidence/'summary.json').read_bytes())
bound=json.loads((a.evidence/'bound_values.json').read_bytes())
freeze=json.loads((a.evidence/'freeze.json').read_bytes())
assert len(rows)==100 and len(bound)==2000
def quantile(v,q):
    x=sorted(v);pos=(len(x)-1)*q;i=int(pos);return x[i]+(x[min(i+1,len(x)-1)]-x[i])*(pos-i)
def stats(v):
    return dict(count=len(v),mean=statistics.mean(v),median=statistics.median(v),min=min(v),max=max(v),p05=quantile(v,.05),p95=quantile(v,.95)) if v else dict(count=0)
errors=[]
def compare(actual,expected):
    assert actual.keys()==expected.keys()
    for k in actual:
        if isinstance(actual[k],dict):compare(actual[k],expected[k])
        elif isinstance(actual[k],(int,float)):
            errors.append(abs(actual[k]-expected[k]));assert math.isclose(actual[k],expected[k],rel_tol=1e-12,abs_tol=1e-12)
        else:assert actual[k]==expected[k]
for r in rows:
    for m in ['psnr','ssim']:assert abs(float(r['delta_'+m])-(float(r['oracle_'+m])-float(r['t028_'+m])))<1e-12
for k,v in summary['metrics'].items():compare(stats([float(r[k]) for r in rows]),v)
compare(stats([float(r['seconds']) for r in rows]),summary['runtime'])
for k,v in summary['histograms'].items():assert dict(Counter(r[k] for r in rows))==v
for m,v in summary['win_equal_loss'].items():
    x=[float(r['delta_'+m]) for r in rows]
    assert dict(win=sum(z>0 for z in x),equal=sum(z==0 for z in x),loss=sum(z<0 for z in x))==v
for ch,regions in summary['bounds_and_gains'].items():
    for region,groups in regions.items():
        for group,expected in groups.items():
            b=[r for r in bound if r['channel']==ch and (region=='all' or r['region']==region) and (group=='all' or r['active']==(group=='active'))]
            actual=dict(**stats([r['value'] for r in b]),lower_hits=sum(r['lower_hit'] for r in b),upper_hits=sum(r['upper_hit'] for r in b))
            compare(actual,expected)
for r in freeze['rows']:
    for name in ['histories.pt','oracle_states.json']:
        f=a.evidence/f"{r['index']:03d}"/name
        assert hashlib.sha256(f.read_bytes()).hexdigest()==r['files'][name]
d=[float(r['delta_psnr']) for r in rows];mean=statistics.mean(d);median=statistics.median(d)
c='substantial chromatic action headroom' if mean>=1.5 and median>=1 else ('limited chromatic action headroom' if mean<.5 else 'mixed chromatic action headroom')
assert c==summary['classification']
receipt=dict(rows=100,retained_artifact_hashes_verified=200,all_summaries_recomputed=True,max_abs_error=max(errors),classification=c)
(a.evidence.parent/'T034A_local_verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
