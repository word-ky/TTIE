import torch
from research_log.T059Q.core import decisions,summarize,classify,LABELS
from research_log.T059Q.storage import tensor_only

def rows():return {'global':[9,7,3,20,21,30],'bank':[1,1,1,2,2,3],'state':[2,0,1,0,1,0],'image':[1,1,1,2,2,3],'E':[-1,0,-1,0,-1,0],'M':[-1,0,-1,-1,0,0]}
def test_tie_state_then_global():
 r=rows();table,c=decisions(r);assert table[0]['E_global']==3;r['state'][0]=1;table,c=decisions(r);assert table[0]['E_global']==3

def test_disagreement_abstains_singleton_not_coverage():
 table,c=decisions(rows());assert table[1]['Q_state']==0 and table[1]['abstained'];assert table[2]['Q_state']==0 and table[2]['singleton'];assert c==dict(non_singleton_banks=2,agreed_non_singleton=1,singleton_banks=1,fraction=.5)

def test_targets_cannot_change_predictions_or_decisions(tmp_path):
 r=rows();out=[]
 for i in range(2):
  path=tmp_path/f'mixed{i}.pt';torch.save(dict(p=torch.tensor(r['E']),delta_t=torch.full((6,),float(i*100000))),path);v,receipt=tensor_only(path,'p');assert receipt['key']=='p';q=dict(r,E=v.tolist());out.append(decisions(q))
 assert out[0]==out[1]

def test_metrics_and_first_applicable_gate():
 s=summarize([dict(regret=0,harm_vs_anchor=0),dict(regret=2,harm_vs_anchor=1)]);assert s['mean_regret']==1 and s['median_regret']==1 and s['p90_regret']==1.8 and s['harm_count']==1 and s['exact_oracle_hit_rate']==.5
 q=dict(s,mean_regret=.5,max_regret=1,p90_regret=1);assert classify(.49,s,q)[0]==LABELS[0];assert classify(.5,s,q)[0]==LABELS[2];assert classify(1,s,s)[0]==LABELS[1];q['harm_count']=2;assert classify(1,s,q)[0]==LABELS[1]
