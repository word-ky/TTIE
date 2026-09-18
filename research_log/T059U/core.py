from research_log.T059S.core import *
import numpy as np
TAU=0.031453661388567547
C2=ROOT/'runs/20260918-002953-ttie-t059c2-holdout/artifacts/T059C2/split.json'
def gate(g):return float(g.double().norm())>=TAU

def classify(coverage,harms,recall,retention,gated_harms,mean,acted_median):
    if coverage<.75:return 'low-norm abstention is too indiscriminate'
    if harms==0:return 'low-norm safety mechanism is not testable on this cohort'
    if recall<.8 or retention<.8 or gated_harms>1:return 'fixed low-norm abstention does not transfer'
    if mean>=0 or acted_median>=0:return 'fixed low-norm abstention does not transfer'
    return 'fixed low-norm abstention is supported as a source-only outer safety candidate'

def summary(rows):
    policies={}
    for name in ['ungated','gated']:
        values=np.array([r[name+'_A'] for r in rows]);acted=np.array([r['acted'] if name=='gated' else True for r in rows]);v=values[acted]
        policies[name]=dict(action_coverage=float(acted.mean()),acted=int(acted.sum()),improved=int((values<0).sum()),harmed=int((values>0).sum()),tied=int((values==0).sum()),mean_A=float(values.mean()),median_A=float(np.median(values)),p90_A=float(np.quantile(values,.9)),max_harm=float(max(0,values.max())),median_A_acted=float(np.median(v)) if len(v) else None)
    harms=[r for r in rows if r['ungated_A']>0];benefits=[r for r in rows if r['ungated_A']<0]
    recall=sum(not r['acted'] for r in harms)/len(harms) if harms else None;retention=sum(r['acted'] for r in benefits)/len(benefits) if benefits else 0.
    g=policies['gated'];label=classify(g['action_coverage'],len(harms),recall,retention,g['harmed'],g['mean_A'],g['median_A_acted'])
    return dict(policies=policies,harmful_anchor_recall=recall,beneficial_action_retention=retention,ungated_harmful_anchors=harms,classification=label,tau=TAU)
