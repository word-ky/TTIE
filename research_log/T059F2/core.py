import torch
from research_log.T059D.core import spearman,distribution,LIMIT

def audit(v):
 p,t=v['p'],v['t'];loss=torch.empty_like(p);banks=[]
 for bi in dict.fromkeys(v['bank_indices'].tolist()):
  ids=(v['bank_indices']==bi).nonzero().flatten();anchor=ids[v['state_indices'][ids]==0];assert len(anchor)==1
  rp=p[ids]-p[anchor.item()];rt=t[ids]-t[anchor.item()]
  assert torch.equal(rp,v['delta_p'][ids]) and torch.equal(rt,v['delta_t'][ids])
  den=torch.dot(rp,rp);num=torch.dot(rp,rt);q=float(num/den) if den>0 else None
  category='scale_unidentified_degenerate' if den==0 else ('positive_scale' if q>0 else 'positive_scale_boundary')
  # Zero is used only to evaluate the continuous loss limit, never for ordering.
  loss_prediction=q*rp if category=='positive_scale' else torch.zeros_like(rp)
  losses=torch.nn.functional.huber_loss(loss_prediction,rt,reduction='none');loss[ids]=losses
  rho,status=spearman(p[ids],t[ids]);choice=int(p[ids].argmin());regret=float(t[ids][choice]-t[ids].min())
  banks.append(dict(bank_index=bi,rows=len(ids),anchor_global_index=int(v['global_indices'][anchor.item()]),numerator=float(num),denominator=float(den),unconstrained_ratio=q,category=category,admissible_scale=q if category=='positive_scale' else None,limit_huber=float(losses.mean()),spearman=rho,spearman_status=status,argmin_regret=regret,predicted_argmin_state_index=int(v['state_indices'][ids[choice]]),ordering_basis='original prediction; positive scalar preserves all pairwise signs and ties'))
 h=float(loss.mean())
 return dict(rows=len(p),bank_count=len(banks),relative_huber=float(torch.nn.functional.huber_loss(v['delta_p'],v['delta_t'])),limit_huber=h,threshold=LIMIT,threshold_margin=LIMIT-h,classification='T059-E value failure is consistent with bankwise positive-scale miscalibration after offset removal' if h<=LIMIT else 'T059-E value failure is not explained by bankwise positive-scale miscalibration after offset removal',partition={c:[b['bank_index'] for b in banks if b['category']==c] for c in ['positive_scale','positive_scale_boundary','scale_unidentified_degenerate']},spearman_distribution=distribution([b['spearman'] for b in banks]),regret_distribution=distribution([b['argmin_regret'] for b in banks]),banks=banks,worst10=sorted(banks,key=lambda b:(-b['limit_huber'],b['bank_index']))[:10])
