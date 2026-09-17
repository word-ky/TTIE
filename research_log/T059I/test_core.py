import torch
from research_log.T059I.core import five,tensor_only,classify,LABELS

def test_distinct_donors_and_tie_order():
 x=torch.zeros(7,1);images=torch.tensor([1,1,2,3,4,5,6]);g=torch.arange(7);r=five(torch.zeros(1,1),x,torch.tensor([1]),images,g,loo=True,device='cpu')
 assert r['global_index'].tolist()==[[2,3,4,5,6]] and len(r['image'][0].unique())==5
 r=five(torch.zeros(1,1),x,torch.tensor([9]),images,g,device='cpu');assert r['global_index'].tolist()==[[0,2,3,4,5]] and r['within_image_tie_count'][0,0]==2

def test_selected_scalar_read(tmp_path):
 p=tmp_path/'values.pt';torch.save({'train':{'target':torch.tensor([1.,2.])},'heldout':{'target':torch.tensor([999.])}},p);t,a=tensor_only(p,'train','target');assert t.tolist()==[1.,2.] and a['bytes']==8

def test_classification_order():
 assert classify(1,0)==LABELS[0] and classify(0,0)==LABELS[1] and classify(0,1)==LABELS[2]
