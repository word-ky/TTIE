import torch
from research_log.T059D.core import distribution
LABELS=['28-D features do not show cross-image local target consistency even within inner-train support under fixed 1-NN','T059-E failure is consistent with an inner-held feature-support shift under fixed 1-NN','28-D local feature support is adequate under fixed 1-NN; parametric head/function fitting remains the primary suspect']
def nearest(query,train,query_images,train_images,global_train,loo=False,device='cuda:0'):
 # Canonical ascending candidate order makes torch.min choose smallest global ID on ties.
 assert torch.equal(global_train,global_train.sort().values)
 tr=train.to(device,dtype=torch.float64);ti=train_images.to(device);neighbors=[];distances=[];ties=[]
 for start in range(0,len(query),64):
  q=query[start:start+64].to(device,dtype=torch.float64)
  d=(q[:,None,:]-tr[None,:,:]).square().sum(-1)
  if loo:d.masked_fill_(query_images[start:start+64].to(device)[:,None]==ti[None,:],float('inf'))
  dist,idx=d.min(1);assert torch.isfinite(dist).all()
  neighbors.append(idx.cpu());distances.append(dist.cpu());ties.append((d==dist[:,None]).sum(1).cpu())
 return dict(index=torch.cat(neighbors),global_index=global_train[torch.cat(neighbors)],distance=torch.cat(distances),tie_count=torch.cat(ties))
def metrics(pred,target,pgrad,truth):
 mask=truth.double().norm(dim=1)>1e-12;pmask=pgrad.double().norm(dim=1)>1e-12
 pgrad=torch.where(pmask[:,None],pgrad,torch.zeros_like(pgrad));dot=(pgrad*truth).sum(1);den=pgrad.norm(dim=1)*truth.norm(dim=1);cos=dot/torch.where(den>0,den,torch.ones_like(den))
 return dict(rows=len(pred),relative_huber=float(torch.nn.functional.huber_loss(pred,target)),detail_eligible=int(mask.sum()),detail_ineligible=int((~mask).sum()),neighbor_noneligible_on_eligible_query=int((mask&~pmask).sum()),detail_positive=float((dot[mask]>0).float().mean()) if mask.any() else None,detail_median=float(torch.quantile(cos[mask],.5)) if mask.any() else None)
def summarize(pred,target,pgrad,truth,banks,images,distances):
 result=metrics(pred,target,pgrad,truth);result['nearest_squared_distance']=distribution(distances.tolist())
 for kind,ids in [('bank',banks),('image',images)]:
  groups=[]
  for i in dict.fromkeys(ids.tolist()):
   m=ids==i;groups.append(dict(id=i,**metrics(pred[m],target[m],pgrad[m],truth[m]),nearest_squared_distance=distribution(distances[m].tolist())))
  result['per_'+kind]=groups
  result[kind+'_metric_distributions']={k:distribution([g[k] for g in groups]) for k in ['relative_huber','detail_positive','detail_median']}
 return result
def gates(s):return dict(value=s['relative_huber']<=.07650849781930447,detail_positive=s['detail_positive']>=.75,detail_median=s['detail_median']>=.5)
def classify(train,held):
 t=gates(train);h=gates(held);return dict(train_gates=t,heldout_gates=h,classification=LABELS[0] if not all(t.values()) else LABELS[1] if not all(h.values()) else LABELS[2])
