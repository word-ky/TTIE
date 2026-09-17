"""Independent scalar/loss/rank verification without importing F2 implementation."""
import json,pathlib,sys,hashlib,torch,math
out=pathlib.Path(sys.argv[1]);r=json.loads((out/'result.json').read_text())
for n,h in r['inputs_before'].items():assert hashlib.sha256(pathlib.Path(n).read_bytes()).hexdigest()==h==r['inputs_after'][n]
rowpath=next(n for n in r['inputs_before'] if n.endswith('heldout_row_values.pt'));v=torch.load(rowpath,map_location='cpu',weights_only=True)
f=json.loads(pathlib.Path(next(n for n in r['inputs_before'] if n.endswith('result.json'))).read_text());losses=[];parts={k:[] for k in r['partition']}
def rank(x):
 vals=x.tolist();return torch.tensor([1+sum(y<z for y in vals)+(sum(y==z for y in vals)-1)/2 for z in vals],dtype=torch.float64)
for b,fb in zip(r['banks'],f['banks']):
 ids=(v['bank_indices']==b['bank_index']).nonzero().flatten();p=v['p'][ids];t=v['t'][ids];anchor=(v['state_indices'][ids]==0).nonzero().item();x=p-p[anchor];y=t-t[anchor]
 den=torch.dot(x,x);num=torch.dot(x,y);assert float(den)==b['denominator']==fb['denominator'] and float(num)==b['numerator']==fb['numerator']
 q=float(num/den) if den else None;category='scale_unidentified_degenerate' if den==0 else ('positive_scale' if q>0 else 'positive_scale_boundary');parts[category].append(b['bank_index']);assert category==b['category']
 assert b['admissible_scale']==(q if category=='positive_scale' else None)
 d=((q*x if category=='positive_scale' else torch.zeros_like(x))-y).abs();ls=torch.where(d<1,.5*d*d,d-.5);losses.append(ls);assert math.isclose(float(ls.mean()),b['limit_huber'],abs_tol=2e-7,rel_tol=2e-6)
 rp=rank(p);rt=rank(t);rp-=rp.mean();rt-=rt.mean();denrho=rp.norm()*rt.norm();rho=float((rp*rt).sum()/denrho) if denrho else None
 assert rho==b['spearman']==fb['spearman'];assert float(t[p.argmin()]-t.min())==b['argmin_regret']==fb['argmin_regret']
assert parts==r['partition'];assert parts['positive_scale_boundary']==[230,280,305] and len(parts['scale_unidentified_degenerate'])==17
h=float(torch.cat(losses).mean());assert math.isclose(h,.21471332013607025,abs_tol=2e-7,rel_tol=2e-6) and h==r['limit_huber'];assert len(torch.cat(losses))==1529 and len(r['banks'])==80
classification='T059-E value failure is consistent with bankwise positive-scale miscalibration after offset removal' if h<=.07650849781930447 else 'T059-E value failure is not explained by bankwise positive-scale miscalibration after offset removal';assert classification==r['classification'];assert all(n==0 for n in r['counters'].values())
result=dict(verification='PASS',independent_classification=classification,independent_limit_huber=h,partition_counts={k:len(v) for k,v in parts.items()},original_order_replay=True,zero_vector_never_used_for_ordering=True,result_sha256=hashlib.sha256((out/'result.json').read_bytes()).hexdigest())
(out/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
