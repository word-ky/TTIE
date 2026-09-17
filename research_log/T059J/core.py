import torch
LIMIT=.07650849781930447
LABELS=['five-donor local scalar support is inadequate even on inner-train LOO under the frozen neighborhood','accurate scalar support exists inside the frozen five-donor neighborhood; donor selection/conditioning, not local support absence, is the primary unresolved failure','even oracle choice among the frozen five donors cannot recover held scalar geometry; local 28-D scalar support is inadequate on inner-held images']
def classify(train,held):return LABELS[0] if train>LIMIT else LABELS[1] if held<=LIMIT else LABELS[2]
def distribution(v):
 x=v.double();return dict(rows=len(x),minimum=float(x.min()),mean=float(x.mean()),p10=float(x.quantile(.1)),median=float(x.quantile(.5)),p90=float(x.quantile(.9)),maximum=float(x.max()))
def floor(values,target):
 assert values.shape==(len(target),5);loss=torch.nn.functional.huber_loss(values,target[:,None].expand_as(values),reduction='none');best,rank=loss.min(1);ordered=values.sort(1).values;span=ordered[:,-1]-ordered[:,0];iqr=ordered[:,3]-ordered[:,1]
 result=dict(rows=len(target),oracle_floor_huber=float(best.mean()),rank1_fraction=float((rank==0).double().mean()),rank_histogram=torch.bincount(rank,minlength=5).tolist(),target_in_donor_range_fraction=float(((target>=ordered[:,0])&(target<=ordered[:,-1])).double().mean()),donor_range_distribution=distribution(span),donor_iqr_distribution=distribution(iqr),tie_rows=int(((loss==best[:,None]).sum(1)>1).sum()))
 return result,dict(best_rank=rank+1,best_loss=best,all_donor_losses=loss)
