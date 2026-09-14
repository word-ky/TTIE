"""Independent scalar summary replay and compact artifact hash check."""
import argparse,csv,json,hashlib,statistics,math
from pathlib import Path
from collections import Counter
p=argparse.ArgumentParser();p.add_argument('result',type=Path);a=p.parse_args();out=a.result/'audit'
rows=list(csv.DictReader((out/'metrics.csv').open()));s=json.loads((out/'summary.json').read_bytes());b=json.loads((out/'bound_values.json').read_bytes());f=json.loads((out/'freeze.json').read_bytes())
assert len(rows)==100 and len(b)==2400
errors=[]
def quant(v,q):
    v=sorted(v);x=(len(v)-1)*q;i=int(x);return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(x-i)
def stats(v):return dict(count=len(v),mean=statistics.mean(v),median=statistics.median(v),min=min(v),max=max(v),p05=quant(v,.05),p95=quant(v,.95)) if v else dict(count=0)
def compare(x,y):
    assert x.keys()==y.keys()
    for k in x:errors.append(abs(x[k]-y[k]));assert math.isclose(x[k],y[k],rel_tol=1e-12,abs_tol=1e-12)
for r in rows:
    for m in ['psnr','ssim']:assert abs(float(r['delta_'+m])-(float(r['common_'+m])-float(r['baseline_'+m])))<1e-12
for k,v in s['metrics'].items():compare(stats([float(r[k]) for r in rows]),v)
for method,channels in s['bounds_and_gains'].items():
    for ch,regions in channels.items():
        for region,groups in regions.items():
            for group,expected in groups.items():
                chosen=[r for r in b if r['method']==method and r['channel']==ch and (region=='all' or r['region']==region) and (group=='all' or r['active']==(group=='active'))]
                compare(dict(**stats([r['value'] for r in chosen]),lower_hits=sum(r['lower_hit'] for r in chosen),upper_hits=sum(r['upper_hit'] for r in chosen)),expected)
for method,hist in s['selected_step_histograms'].items():assert dict(Counter(r[method+'_step'] for r in rows))==hist
for m,expected in s['win_equal_loss'].items():
    v=[float(r['delta_'+m]) for r in rows];assert dict(win=sum(x>0 for x in v),equal=sum(x==0 for x in v),loss=sum(x<0 for x in v))==expected
for r in f['rows']:
    for method,record in r['methods'].items():
        for n,h in record['files'].items():
            if n!='output.pt':assert hashlib.sha256((out/f'{r["index"]:03d}'/method/n).read_bytes()).hexdigest()==h['sha256']
classification='materially positive' if statistics.mean(float(r['delta_psnr']) for r in rows)>=.3 and statistics.mean(float(r['delta_ssim']) for r in rows)>=0 else 'negative/insufficient'
assert classification==s['classification']
receipt=dict(rows=100,compact_hashes_verified=600,all_summaries_recomputed=True,max_abs_error=max(errors),classification=classification)
(a.result/'T036A_local_verification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
