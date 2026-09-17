"""Separate result verifier; does not import the diagnostic implementation."""
import json,pathlib,sys,hashlib,math,torch
out=pathlib.Path(sys.argv[1]);r=json.loads((out/'result.json').read_text());root=pathlib.Path(r['input_directory'])
for n,h in r['inputs_before'].items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h==r['inputs_after'][n]
v=torch.load(root/'heldout_row_values.pt',map_location='cpu',weights_only=True)
losses=[];failed=[]
def rank(x):
 vals=x.tolist();return torch.tensor([1+sum(y<z for y in vals)+(sum(y==z for y in vals)-1)/2 for z in vals],dtype=torch.float64)
def rho(x,y):
 if len(x)<2:return None
 x=rank(x);y=rank(y);x-=x.mean();y-=y.mean();d=x.norm()*y.norm()
 return float((x*y).sum()/d) if d else None
for b in r['banks']:
 ids=(v['bank_indices']==b['bank_index']).nonzero().flatten();p=v['p'][ids];t=v['t'][ids];anchor=(v['state_indices'][ids]==0).nonzero().item()
 x=p-p[anchor];y=t-t[anchor];den=torch.dot(x,x);a=max(0.,float(torch.dot(x,y)/den)) if den else 0.;assert a==b['scale']
 z=x*a;d=(z-y).abs();ls=torch.where(d<1,.5*d*d,d-.5);losses.append(ls)
 assert math.isclose(float(ls.mean()),b['corrected_huber'],abs_tol=2e-7,rel_tol=2e-6)
 before=rho(p,t);after=rho(z,y);reg=float(t[p.argmin()]-t.min());reg2=float(t[z.argmin()]-t.min())
 assert before==b['spearman'] and after==b['scaled_spearman'] and reg==b['argmin_regret'] and reg2==b['scaled_argmin_regret']
 if before!=after or reg!=reg2:failed.append(b['bank_index'])
h=float(torch.cat(losses).mean());assert math.isclose(h,r['corrected_huber'],abs_tol=2e-7,rel_tol=2e-6);assert failed==r['invariance_failures']
limit=.07650849781930447
classification=None if failed else ('T059-E value failure is consistent with bankwise positive-scale miscalibration after offset removal' if h<=limit else 'T059-E value failure is not explained by bankwise positive-scale miscalibration after offset removal')
assert classification==r['classification'];assert len(torch.cat(losses))==1529 and len(r['banks'])==80;assert all(x==0 for x in r['counters'].values())
e=json.loads((root/'heldout_statistics.json').read_text());assert float(torch.nn.functional.huber_loss(v['delta_p'],v['delta_t']))==e['bank_relative_huber']==r['relative_huber']
result=dict(verification='PASS',independent_corrected_huber=h,independent_classification=classification,invariance_failures=failed,scientific_status=r['status'],directional_metrics='SHA-bound evidence readback only, not fresh reconstruction',result_sha256=hashlib.sha256((out/'result.json').read_bytes()).hexdigest())
(out/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
