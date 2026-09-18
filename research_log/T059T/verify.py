"""Independent scalar-loop dot products, Adam moment algebra, ranks and gates."""
import argparse,json,math
from pathlib import Path
from research_log.T059T.run import sha,write,utc

def rank(values):return [1+sum(y<x for y in values)+(sum(y==x for y in values)-1)/2 for x in values]
def correlation(x,y):
    a=rank(x);b=rank(y);am=sum(a)/len(a);bm=sum(b)/len(b)
    return math.fsum((v-am)*(w-bm) for v,w in zip(a,b))/math.sqrt(math.fsum((v-am)**2 for v in a)*math.fsum((w-bm)**2 for w in b))
def quantile(values,q):
    s=sorted(values);p=(len(s)-1)*q;i=math.floor(p);j=math.ceil(p);return s[i]+(s[j]-s[i])*(p-i)
def close(a,b):assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-20),(a,b)
def main(out):
    source=json.loads((out/'frozen_numeric_inputs.json').read_bytes());table=json.loads((out/'diagnostic_table.json').read_bytes());result=json.loads((out/'result.json').read_bytes());receipt=json.loads((out/'input_receipt.json').read_bytes());values=[]
    assert len(source)==80
    for s in source:
        g=s['g'];v=s['v'];ref=s['reference'];norm=math.sqrt(math.fsum(x*x for x in ref));eligible=norm>1e-12;assert eligible==s['eligible']
        if not eligible:continue
        # Explicit fresh Adam moment/bias corrections; no optimizer is instantiated.
        replay=[]
        for x in g:
            m=(1-.9)*x;second=(1-.999)*x*x;replay.append(-.05*(m/(1-.9))/(math.sqrt(second/(1-.999))+1e-8))
        error=max(abs(a-b) for a,b in zip(replay,v));assert error<1e-7
        L=math.fsum(a*b for a,b in zip(ref,v));A=s['mse1']-s['mse0'];active=[i for i,gval in enumerate(g) if abs(gval)>1e-8]
        d=dict(L=L,A=A,R=A-L,predicted_gradient_norm=math.sqrt(math.fsum(x*x for x in g)),reference_gradient_norm=norm,step_norm=math.sqrt(math.fsum(x*x for x in v)),saturation_fraction=sum(abs(v[i])/.05>=.9 for i in active)/len(active),linear_descent=L<0,actual_harm=A>0,overshoot_flip=L<0 and A>0)
        t=table[len(values)];assert all(t[k]==s[k] for k in ['index','bank_index','state_index','image_id'])
        for k,x in d.items():
            if isinstance(x,bool):assert x==t[k]
            else:close(x,t[k])
        values.append(d)
    assert len(values)==61
    linear=sum(r['linear_descent'] for r in values);harm=sum(r['actual_harm'] for r in values);flip=sum(r['overshoot_flip'] for r in values)
    assert (linear,harm,flip)==(result['linear_descent'],result['actual_harm'],result['overshoot_flip'])
    for k,summary in result['statistics'].items():
        for name,q in [('median',.5),('p10',.1),('p90',.9)]:close(quantile([r[k] for r in values],q),summary[name])
    for k,c in result['spearman_with_step_norm'].items():close(correlation([r[k] for r in values],[r['step_norm'] for r in values]),c)
    if linear/61<.9:label='transferred detail direction itself is insufficient; close the direct detail-step branch'
    elif flip/harm<.5 or quantile([r['saturation_fraction'] for r in values],.5)<.8:label='optimizer-scale/curvature mismatch not established; close the direct detail-step branch'
    else:label='optimizer-scale/curvature mismatch is supported as a source-only diagnosis'
    assert label==result['classification']
    for n,h in receipt['input_hashes_before'].items():assert sha(n)==h
    for n,h in receipt['source_bindings'].items():assert sha(n)==h
    write(out/'verification.json',dict(status='PASS',alignment=80,eligible=61,harmful=harm,all_fields_and_summary_and_classification=True,independent='Python scalar moment algebra/dot/norm, pairwise average ranks and interpolated order-statistic quantiles',utc=utc()));print('INDEPENDENT_PASS80/61',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
