import numpy as np,pytest,torch
from research_log.T059R.core import ranks,spearman,descriptors,auroc,group_summary,classify,NEG,POS
from research_log.T059R.storage import tensor_only

def test_average_ranks_and_undefined():
 assert ranks([3,1,1,2]).tolist()==[4,1.5,1.5,3];assert spearman([1,1,2],[3,3,1])==-1;assert spearman([1,1],[1,2]) is None

def rows():return {'global':[1,2,3,4,5],'bank':[2,2,1,1,3],'state':[0,1,0,1,0],'image':[1,1,2,2,3],'E':[1,0,1,0,0],'M':[0,1,0,1,0]}
def test_rank_ties_singletons_and_constant_block():
 r=rows();t=descriptors(r);assert len(t)==2 and t[0]['bank']==1 and t[0]['uncertainty_rank']==1 and t[1]['uncertainty_rank']==2;assert t[0]['E_state']==1;r['M'][0]=1
 with pytest.raises(ValueError,match='undefined'):descriptors(r)

def test_auroc_summaries_and_conditions():
 assert auroc([1,1,0],[True,False,False])==.75;assert group_summary([0,1,2,3])==dict(count=4,median=1.5,q25=.75,q75=2.25,iqr=1.5)
 t=[dict(bank=1,unsafe_E=True,harm_vs_anchor=2,uncertainty_percentile=0),dict(bank=2,unsafe_E=True,harm_vs_anchor=1,uncertainty_percentile=20),dict(bank=3,unsafe_E=False,harm_vs_anchor=0,uncertainty_percentile=100)];assert classify(t,.9)[0]==POS;assert classify(t,.89)[0]==NEG;t[0]['uncertainty_percentile']=11;assert classify(t,.99)[0]==NEG;t[0]['uncertainty_percentile']=0;t[1]['uncertainty_percentile']=26;assert classify(t,.99)[0]==NEG

def test_target_replacement_cannot_change_descriptors(tmp_path):
 r=rows();out=[]
 for i in range(2):
  f=tmp_path/f'mixed{i}.pt';torch.save(dict(p=torch.tensor(r['E'],dtype=torch.float32),delta_t=torch.full((5,),float(i*999))),f);v,a=tensor_only(f,'p');out.append(descriptors(dict(r,E=v.tolist())));assert a['key']=='p'
 assert out[0]==out[1]
