"""Post-freeze threshold-free support/validity relation; no selection."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata,spearmanr
def classify(auc,rho,valid_median,invalid_median):
    if auc>=.7 and rho<=-.3 and invalid_median>valid_median:return 'promising source-support proxy'
    if auc>=.6 or rho<=-.2:return 'weak/mixed support relation'
    return 'source-support hypothesis not supported'
def main():
    p=argparse.ArgumentParser();p.add_argument('--support',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);a=p.parse_args()
    scores=json.loads((a.support/'scores.json').read_bytes());labels=json.loads((a.reference/'validity.json').read_bytes())
    assert len(scores)==len(labels)==4100
    for s,r in zip(scores,labels):assert (s['index'],s['step'])==(r['index'],r['step'])
    d=np.array([s['distance'] for s in scores]);invalid=np.array([not r['valid'] for r in labels]);n=int(invalid.sum());m=len(d)-n
    auc=float((rankdata(d)[invalid].sum()-n*(n+1)/2)/(n*m))
    independent_auc=float(sum(np.sum(x>d[~invalid])+.5*np.sum(x==d[~invalid]) for x in d[invalid])/(n*m))
    assert abs(auc-independent_auc)<=1e-12
    ok=np.array([r['cosine'] is not None for r in labels]);c=np.array([r['cosine'] if r['cosine'] is not None else np.nan for r in labels])
    rho=float(spearmanr(d[ok],c[ok]).statistic);rho2=float(np.corrcoef(rankdata(d[ok]),rankdata(c[ok]))[0,1]);assert abs(rho-rho2)<=1e-12
    def dist(x):return dict(count=len(x),median=float(np.median(x)),q25=float(np.quantile(x,.25)),q75=float(np.quantile(x,.75)),iqr=float(np.quantile(x,.75)-np.quantile(x,.25)))
    def group(mask):return dict(count=int(mask.sum()),distance_median=float(np.median(d[mask])),invalid_fraction=float(invalid[mask].mean()),cosine_median=float(np.nanmedian(c[mask])))
    valid_stats=dist(d[~invalid]);invalid_stats=dist(d[invalid])
    result=dict(auc_invalid=auc,spearman_rho=rho,valid=valid_stats,invalid=invalid_stats,
        degenerate_pairs=int((~ok).sum()),degenerate_fraction=float((~ok).mean()),
        by_step={str(t):group(np.array([s['step']==t for s in scores])) for t in [0,10,20,30,40]},
        original_selected=group(np.array([s['original_selected'] for s in scores])),
        classification=classify(auc,rho,valid_stats['median'],invalid_stats['median']),
        independent_auc_abs_error=abs(auc-independent_auc),independent_rho_abs_error=abs(rho-rho2))
    (a.reference/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
