import torch
from research_log.T059J.core import floor,classify,LABELS

def test_oracle_tie_frozen_order_and_coverage():
 r,v=floor(torch.tensor([[1.,-1.,2.,3.,4.],[9.,8.,7.,6.,5.]]),torch.tensor([0.,5.]))
 assert v['best_rank'].tolist()==[1,5] and r['rank_histogram']==[1,0,0,0,1] and r['oracle_floor_huber']==.25 and r['target_in_donor_range_fraction']==1
 assert r['donor_iqr_distribution']['median']==2

def test_classification_priority():
 assert classify(1,0)==LABELS[0] and classify(0,0)==LABELS[1] and classify(0,1)==LABELS[2]
