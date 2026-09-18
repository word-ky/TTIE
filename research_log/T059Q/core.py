import numpy as np
LABELS=['dual-head consensus is too abstaining to be a useful safety candidate; stop','exact dual-head agreement is not a sufficient target-free safety gate; stop','dual-head disagreement is supported as a target-free safety-gate candidate on the already-opened inner-held source diagnostic']
def decisions(rows):
 table=[]
 for bank in sorted(set(rows['bank'])):
  ids=[i for i,b in enumerate(rows['bank']) if b==bank];anchor=[i for i in ids if rows['state'][i]==0];assert len(anchor)==1;anchor=anchor[0]
  choices={h:min(ids,key=lambda i:(rows[h][i],rows['state'][i],rows['global'][i])) for h in ['E','M']};agree=choices['E']==choices['M'];q=choices['E'] if agree else anchor
  table.append(dict(bank=int(bank),image=int(rows['image'][anchor]),rows=len(ids),singleton=len(ids)==1,anchor=anchor,anchor_global=int(rows['global'][anchor]),a_E=choices['E'],a_M=choices['M'],a_Q=q,agreement=agree,abstained=not agree,**{h+'_state':int(rows['state'][i]) for h,i in [('E',choices['E']),('M',choices['M']),('Q',q)]},**{h+'_global':int(rows['global'][i]) for h,i in [('E',choices['E']),('M',choices['M']),('Q',q)]}))
 eligible=[b for b in table if not b['singleton']];coverage=dict(non_singleton_banks=len(eligible),agreed_non_singleton=sum(b['agreement'] for b in eligible),singleton_banks=sum(b['singleton'] for b in table));coverage['fraction']=coverage['agreed_non_singleton']/len(eligible);return table,coverage

def summarize(values):
 regret=np.array([v['regret'] for v in values],dtype=np.float64);harm=np.array([v['harm_vs_anchor'] for v in values],dtype=np.float64)
 return dict(banks=len(values),exact_oracle_hit_count=int((regret==0).sum()),exact_oracle_hit_rate=float((regret==0).mean()),mean_regret=float(regret.mean()),median_regret=float(np.median(regret)),p90_regret=float(np.quantile(regret,.9,method='linear')),max_regret=float(regret.max()),harm_count=int((harm>0).sum()),harm_fraction=float((harm>0).mean()))
def classify(coverage,e,q):
 criteria=dict(lower_mean=q['mean_regret']<e['mean_regret'],lower_maximum=q['max_regret']<e['max_regret'],p90_no_worse=q['p90_regret']<=e['p90_regret'],harm_count_no_increase=q['harm_count']<=e['harm_count'])
 return (LABELS[0] if coverage<.5 else LABELS[1] if not all(criteria.values()) else LABELS[2]),criteria
