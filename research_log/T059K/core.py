import torch
LIMIT=.07650849781930447
LABELS=['global inner-train scalar support is inadequate even on train-LOO under the fixed source pool','marginal scalar support exists in the fixed inner-train pool; the unresolved failure is localization/conditioning of that support rather than absence of scalar values','inner-held scalar targets are inadequately supported even by the global fixed inner-train scalar pool; 28-D locality alone cannot explain the transfer failure']
def classify(t,h):return LABELS[0] if t>LIMIT else LABELS[1] if h<=LIMIT else LABELS[2]
def dist(v):
 x=v.double();return dict(rows=len(x),minimum=float(x.min()),mean=float(x.mean()),p10=float(x.quantile(.1)),median=float(x.quantile(.5)),p90=float(x.quantile(.9)),maximum=float(x.max()))
def oracle(pool,query,target,five,loo=False,device='cuda:0'):
 assert torch.equal(pool['global_index'],pool['global_index'].sort().values);pt=pool['target'].to(device);px=pool['x'].to(device,dtype=torch.float64);pi=pool['image'].to(device);pg=pool['global_index'].to(device);parts={k:[] for k in ['index','global_index','image','loss','candidate_count','in_global_range','in_five','distance','distance_rank','distance_percentile','scalar_minimum_tie_count']}
 for start in range(0,len(target),64):
  y=target[start:start+64].to(device);im=query['image'][start:start+64].to(device);allowed=(im[:,None]!=pi[None,:]) if loo else torch.ones((len(y),len(pt)),dtype=torch.bool,device=device)
  loss=torch.nn.functional.huber_loss(pt[None,:].expand(len(y),-1),y[:,None].expand(-1,len(pt)),reduction='none');loss=loss.masked_fill(~allowed,float('inf'));minimum,idx=loss.min(1);assert torch.isfinite(minimum).all()
  d=(query['x'][start:start+64].to(device,dtype=torch.float64)[:,None,:]-px[None,:,:]).square().sum(-1);chosen=d.gather(1,idx[:,None]).flatten();rank=1+(((d<chosen[:,None])|((d==chosen[:,None])&(pg[None,:]<pg[idx,None])))&allowed).sum(1);count=allowed.sum(1)
  low=pt[None,:].expand(len(y),-1).masked_fill(~allowed,float('inf')).min(1).values;high=pt[None,:].expand(len(y),-1).masked_fill(~allowed,float('-inf')).max(1).values
  values=dict(index=idx,global_index=pg[idx],image=pi[idx],loss=minimum,candidate_count=count,in_global_range=(y>=low)&(y<=high),in_five=(idx[:,None]==five[start:start+len(y)].to(device)).any(1),distance=chosen,distance_rank=rank,distance_percentile=(rank.double()-1)/(count.double()-1),scalar_minimum_tie_count=(loss==minimum[:,None]).sum(1))
  for k,v in values.items():parts[k].append(v.cpu())
 rows={k:torch.cat(v) for k,v in parts.items()};summary=dict(rows=len(target),global_oracle_huber=float(rows['loss'].mean()),target_in_global_scalar_range=float(rows['in_global_range'].double().mean()),frozen_five_membership=float(rows['in_five'].double().mean()),distance_rank=dist(rows['distance_rank']),distance_percentile=dist(rows['distance_percentile']),candidate_count=dist(rows['candidate_count']),oracle_squared_distance=dist(rows['distance']),queries_with_scalar_ties=int((rows['scalar_minimum_tie_count']>1).sum()))
 return summary,rows
