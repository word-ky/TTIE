"""Independent replay of every frozen state lookup and global selection."""
import argparse,json,hashlib
from pathlib import Path
from decimal import Decimal,localcontext

def main(out):
    raw=Path('/media/wenchang/F/wjq/TTIE/runs/T062A-fixed-zr')
    b=(raw/'freeze.json').read_bytes();assert hashlib.sha256(b).hexdigest()=='301c041aedeb7d4925560a860417891e6bc64fa616da3c1c10565b10d3496d00'
    frozen=json.loads(b)['rows'];rows=[json.loads((out/f'{i:03d}.json').read_bytes()) for i in range(100)]
    table=json.loads((out/'table.json').read_bytes());result=json.loads((out/'selection.json').read_bytes())
    assert len(table)==41 and result['table_sha256']==hashlib.sha256((out/'table.json').read_bytes()).hexdigest()
    for i,(r,f) in enumerate(zip(rows,frozen)):
        assert r['index']==f['index']==i and r['low']==f['low'] and r['state_hashes']==f['rendered_hashes']
        assert len(r['psnr'])==len(r['ssim'])==len(r['independent_metric_errors'])==41 and max(r['independent_metric_errors'])<1e-11
    def mean(xs):
        with localcontext() as c:c.prec=80;return float(sum((Decimal.from_float(v) for v in xs),Decimal(0))/100)
    eligible=[];errors=[]
    for k,t in enumerate(table):
        assert t['step']==k
        p=[r['psnr'][k]-r['t036_psnr'] for r in rows];s=[r['ssim'][k]-r['t036_ssim'] for r in rows];b=[r['psnr'][k]-r['t026_psnr'] for r in rows]
        median=(sorted(p)[49]+sorted(p)[50])/2
        for a,z in [(mean(p),t['mean_delta']),(median,t['median_delta']),(mean(s),t['mean_delta_ssim']),(mean([r['psnr'][k] for r in rows]),t['mean_psnr']),(mean([r['ssim'][k] for r in rows]),t['mean_ssim'])]:errors.append(abs(a-z))
        assert t['improve']==sum(v>0 for v in p) and t['regress']==sum(v<0 for v in p) and t['tie']==sum(v==0 for v in p)
        count=sum(v<0 for v in b);worst=min(b);safe=count<=29 and worst>=-5.614 and mean(s)>=-.001
        assert count==t['regressions_t026'] and worst==t['worst_delta_t026'] and safe==t['safety_eligible']
        if safe:eligible.append((k,mean(p),median))
    assert max(errors)<1e-12
    chosen=sorted(eligible,key=lambda x:(-x[1],x[0]))[0] if eligible else None
    assert result['eligible_steps']==[x[0] for x in eligible]
    assert result['k_star']==(chosen[0] if chosen else None)
    passed=chosen is not None and chosen[1]>=2 and chosen[2]>0
    assert result['verdict']==('PASS' if passed else 'NEGATIVE')
    receipt=dict(status='PASS',state_lookups=4100,steps=41,max_aggregate_error=max(errors),max_independent_metric_error=max(max(r['independent_metric_errors']) for r in rows),k_star=result['k_star'],verdict=result['verdict'])
    with (out/'verification.json').open('x') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(receipt))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
