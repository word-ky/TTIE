"""Frozen C2 held-out bank-relative value/gradient audit."""
import argparse,json,time,math
from pathlib import Path
import torch
from ttie.energy_model import load_energy
from ttie.sobolev_train import cosine
from research_log.T059C2.run import initialize,bindings,load_side,statistics,unchanged,ROOT
from research_log.T058A_tangent.core import sha,thash,utc
from research_log.T059A.support import atomic_json
from research_log.T059D.core import bank_metrics,distribution,LIMIT
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();initialize()
source=json.loads(Path('research_log/T059D_source_binding.json').read_bytes())
for n,h in source.items():assert sha(Path(n))==h,n
banks,split,assets,_=bindings();assets.update(source)
c2=ROOT/'runs/20260918-002953-ttie-t059c2-holdout/artifacts/T059C2'
assert sha(c2/'complete.json')=='1fdf43368214e466bbcd007873938a00b4722c7d7f622ac02c3c70f3901086bf'
marker=json.loads((c2/'complete.json').read_bytes());assets[str(c2/'complete.json')]=sha(c2/'complete.json')
for n,h in marker['files'].items():assert sha(c2/n)==h,n;assets[str(c2/n)]=h
assert sha(c2/'head.pt')=='df874a53aadd359c1363e5866179a251dbf5ca86ceb72988d53485873202d93c'
c2split=json.loads((c2/'split.json').read_bytes());assert all(c2split[k]==v for k,v in json.loads(json.dumps(split)).items())
x,mse,leg,detail,access,tensors=load_side(banks,split,'heldout');assert len(x)==1460 and len(split['bank_indices']['heldout'])==80
head=load_energy(c2/'head.pt');headbefore={k:thash(v) for k,v in head.state_dict().items()};actual=statistics(head,x,mse,leg,detail);expected=json.loads((c2/'summary.json').read_bytes())['heldout'];replay={}
for group,key in [('legacy','value_huber'),('legacy','positive_fraction'),('legacy','median_cosine'),('detail','positive_fraction'),('detail','median_cosine')]:
    av=actual[group][key];ev=expected[group][key];error=abs(av-ev);tol=max(2e-6,2e-5*max(abs(av),abs(ev)))
    replay[group+'_'+key]=dict(actual=av,expected=ev,delta=av-ev,absolute_error=error,tolerance=tol,margin=tol-error,passed=math.isclose(av,ev,abs_tol=2e-6,rel_tol=2e-5))
passed=all(v['passed'] for v in replay.values()) and all(actual[g]['direction_rows']==expected[g]['direction_rows'] for g in ['legacy','detail'])
atomic_json(a.out/'replay.json',dict(passed=passed,comparisons=replay,actual=actual,expected=expected,tolerance=dict(atol=2e-6,rtol=2e-5)))
if not passed:
    atomic_json(a.out/'failure.json',dict(classification='T059-D replay mismatch',training_runs=0,optimizer_steps=0,completed_utc=utc()));raise SystemExit('T059-D replay mismatch')
leaf=x.detach().clone().requires_grad_(True);pred=head.standardized(leaf);q,=torch.autograd.grad((pred*head.y_scale+head.y_mean).sum(),leaf)
gl=torch.einsum('bfi,bf->bi',leg['jacobian'],q);gd=torch.einsum('bfi,bf->bi',detail['jacobian'],q);assert thash(gd)==expected['detail']['gradient_sha256']
lcos=cosine(gl,leg['reference_gradient']);dcos=cosine(gd,detail['reference_gradient'])
pred=pred.detach();target=((mse.double()+1e-6).log().float()-head.y_mean)/head.y_scale
indices=split['row_indices']['heldout'];records=[split['canonical'][i] for i in indices];bi=torch.tensor([r['bank_index'] for r in records]);si=torch.tensor([r['state_index'] for r in records]);gi=torch.tensor(indices)
perbank,dp,dt,anchors=bank_metrics(pred,target,lcos,dcos,leg['direction_mask'],detail['direction_mask'],bi,si,gi)
assert len(perbank)==80 and sum(b['rows'] for b in perbank)==1460
for b in perbank:b['image_id']=banks[b['bank_index']]['image_id']
relative=float(torch.nn.functional.huber_loss(dp,dt,delta=1.));unanchored=float(torch.nn.functional.huber_loss(pred,target,delta=1.));assert unanchored==actual['legacy']['value_huber']
classification='C2 value failure is consistent with bankwise additive-offset miscalibration' if relative<=LIMIT else 'C2 value failure is not explained by bankwise additive offsets'
dists={k:distribution([b[k] for b in perbank]) for k in ['unanchored_huber','relative_huber','spearman','argmin_regret','anchor_residual']}
for group in ['legacy','detail']:
 for k in ['positive_fraction','median_cosine']:dists[group+'_'+k]=distribution([b[group][k] for b in perbank])
worst={k:[b['bank_index'] for b in sorted([b for b in perbank if b[k] is not None],key=lambda b:((-b[k]) if descending else b[k],b['bank_index']))[:10]] for k,descending in [('relative_huber',True),('argmin_regret',True),('spearman',False)]}
for group in ['legacy','detail']:
 for k in ['positive_fraction','median_cosine']:worst[group+'_'+k]=[b['bank_index'] for b in sorted([b for b in perbank if b[group][k] is not None],key=lambda b:(b[group][k],b['bank_index']))[:10]]
summary=dict(classification=classification,c2_verdict_unchanged='image-held-out dual-tangent generalization not supported under fixed split',banks=80,rows=1460,anchors=80,unanchored_huber=unanchored,bank_relative_huber=relative,value_huber_limit=LIMIT,gate_margin=LIMIT-relative,distributions=dists,worst_bank_indices=worst,global_directional_statistics_unchanged=actual,tie_rule='Average ranks; singleton or constant ranks undefined/null. Argmin ties choose earliest state_index; diagnostic worst-bank ties choose bank_index ascending.',anchor_rule='Unique state_index==0 in each bank; all1460rows including anchors remain in aggregate',value_convention='log(mse.double()+1e-6).float(), then original head y_mean/y_scale; head.standardized(x); float32 anchor subtraction and Huber delta1')
payload=dict(p=pred,t=target,delta_p=dp,delta_t=dt,anchor_local_indices=anchors,legacy_cosines=lcos,detail_cosines=dcos,legacy_mask=leg['direction_mask'],detail_mask=detail['direction_mask'],bank_indices=bi,state_indices=si,global_indices=gi)
torch.save(payload,a.out/'row_values.pt');atomic_json(a.out/'banks.json',perbank);atomic_json(a.out/'summary.json',summary);atomic_json(a.out/'access.json',access)
assert access['tensor_hashes']=={k:thash(v) for k,v in tensors.items()} and headbefore=={k:thash(v) for k,v in head.state_dict().items()};unchanged(assets)
receipt=dict(classification=classification,source_bindings=source,asset_hashes_before=assets,asset_hashes_after=assets,input_tensor_hashes_before=access['tensor_hashes'],input_tensor_hashes_after={k:thash(v) for k,v in tensors.items()},head_before=headbefore,head_after=headbefore,row_tensor_hashes={k:thash(v) for k,v in payload.items()},head_sha256=sha(c2/'head.pt'),training_runs=0,optimizer_steps=0,new_source_image_opens=0,reference_gradient_recomputations=0,new_feature_forwards=0,target_domain_access=0,lolv2_access=0,official_test_access=0,inference_reference_leakage=0,device='cpu frozen C2 replay',seconds=time.perf_counter()-start,completed_utc=utc())
atomic_json(a.out/'receipt.json',receipt);atomic_json(a.out/'complete.json',dict(classification=classification,files={n:sha(a.out/n) for n in ['replay.json','row_values.pt','banks.json','summary.json','access.json','receipt.json']},completed_utc=utc()));print(json.dumps(summary),flush=True)
