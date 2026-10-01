import numpy as np
NEG='full-curve cross-head disagreement is not a convincing tail-uncertainty mechanism; stop the dual-head disagreement line'
POS='full-curve cross-head disagreement is supported as a source-only uncertainty mechanism candidate'
def ranks(x):
 x=np.asarray(x,dtype=np.float64);order=np.argsort(x,kind='stable');r=np.empty(len(x),dtype=np.float64);i=0
 while i<len(x):
  j=i+1
  while j<len(x) and x[order[j]]==x[order[i]]:j+=1
  r[order[i:j]]=(i+1+j)/2;i=j
 return r

def spearman(x,y):
 a=ranks(x);b=ranks(y);a-=a.mean();b-=b.mean();den=np.sqrt(np.dot(a,a)*np.dot(b,b));return None if den==0 else float(np.dot(a,b)/den)

def descriptors(rows):
 out=[]
 for bank in sorted(set(rows['bank'])):
  ids=sorted([i for i,b in enumerate(rows['bank']) if b==bank],key=lambda i:(rows['state'][i],rows['global'][i]))
  if len(ids)==1:continue
  a=[rows['E'][i] for i in ids];b=[rows['M'][i] for i in ids];rho=spearman(a,b)
  if rho is None:raise ValueError('fixed statistic undefined in bank '+str(bank))
  anchors=[i for i in ids if rows['state'][i]==0];assert len(anchors)==1;choices={h:min(ids,key=lambda i:(rows[h][i],rows['state'][i],rows['global'][i])) for h in ['E','M']}
  out.append(dict(bank=bank,image=rows['image'][ids[0]],rows=len(ids),row_indices=ids,global_ids=[rows['global'][i] for i in ids],state_indices=[rows['state'][i] for i in ids],p_E=a,p_M=b,rho_EM=rho,u=1-rho,anchor=anchors[0],**{'a_'+h:i for h,i in choices.items()},**{h+'_state':rows['state'][i] for h,i in choices.items()}))
 ordered=sorted(out,key=lambda b:(-b['u'],b['bank']))
 for rank,b in enumerate(ordered,1):b['uncertainty_rank']=rank;b['uncertainty_percentile']=100*(rank-1)/(len(out)-1)
 return out

def auroc(u,labels):
 u=np.asarray(u);labels=np.asarray(labels,dtype=bool);a=u[labels];b=u[~labels];return None if len(a)==0 or len(b)==0 else float(((a[:,None]>b[None,:]).astype(float)+.5*(a[:,None]==b[None,:])).mean())
def group_summary(values):
 a=np.asarray(values,dtype=np.float64)
 if len(a)==0:return dict(count=0,median=None,q25=None,q75=None,iqr=None)
 q=np.quantile(a,[.25,.5,.75]);return dict(count=len(a),median=float(q[1]),q25=float(q[0]),q75=float(q[2]),iqr=float(q[2]-q[0]))
def classify(table,auc):
 unsafe=[b for b in table if b['unsafe_E']];worst=max(table,key=lambda b:(b['harm_vs_anchor'],-b['bank']));conditions=dict(auroc_at_least_090=auc is not None and auc>=.9,all_unsafe_top_quartile=all(b['uncertainty_percentile']<=25 for b in unsafe),maximum_harm_top_decile=worst['uncertainty_percentile']<=10)
 label='T059-R inconclusive; too few unsafe banks for this diagnostic' if len(unsafe)<2 else NEG if not all(conditions.values()) else POS
 return label,conditions
