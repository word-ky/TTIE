import torch
LIMIT=.07650849781930447
YES='T059-G scalar failure is consistent with nearest-distance support shift under fixed train-decile reweighting'
NO='nearest-distance shift alone does not explain T059-G scalar failure; conditional scalar mismatch remains'
def boundaries(d):return torch.quantile(d.double(),torch.arange(1,10,dtype=torch.float64)/10,interpolation='linear')
def assign(d,b):return torch.bucketize(d.double(),b,right=True)
def reweight(train_d,held_d,train_loss,held_loss,b):
 tb=assign(train_d,b);hb=assign(held_d,b);bins=[]
 for i in range(10):
  tr=tb==i;he=hb==i;assert tr.any(),'Empty train bin: stop without alternative binning'
  bins.append(dict(bin=i,train_rows=int(tr.sum()),held_rows=int(he.sum()),held_fraction=float(he.sum())/len(held_d),train_mean_huber=float(train_loss[tr].double().mean()),held_mean_huber=float(held_loss[he].double().mean()) if he.any() else None))
 value=sum(x['held_fraction']*x['train_mean_huber'] for x in bins)
 return dict(bins=bins,distance_reweighted_train_huber=value,threshold=LIMIT,threshold_margin=LIMIT-value,classification=YES if value>LIMIT else NO)
def summary(d):
 d=d.double();return dict(rows=len(d),minimum=float(d.min()),mean=float(d.mean()),median=float(d.quantile(.5)),p10=float(d.quantile(.1)),p90=float(d.quantile(.9)),maximum=float(d.max()))
