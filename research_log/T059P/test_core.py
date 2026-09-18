import torch
from research_log.T059P.core import context,nearest,extract_bank,classify,LIMIT,LABELS
from ttie.clip_signal import FrozenCLIP

def test_context_keeps_all_views_and_bank_anchor():
 e=torch.arange(30,dtype=torch.float32).reshape(3,5,2);q=context(e,torch.tensor([1,1,2]),torch.tensor([0,1,0]));assert torch.equal(q['anchor'],torch.tensor([0,0,2]));assert q['z'].shape==(3,20);assert torch.equal(q['z'][1,:10],(e[1]-e[0]).flatten());assert torch.equal(q['z'][1,10:],e[0].flatten())

def test_ties_and_image_exclusion():
 x=torch.tensor([[0.],[0.],[2.]]);im=torch.tensor([1,2,3]);ids=torch.tensor([3,7,9]);a=nearest(x,x,im,im,ids,loo=True,device='cpu');assert a['global_index'].tolist()==[7,3,3];assert a['tie_count'].tolist()==[1,1,2];b=nearest(x[2:],x,torch.tensor([4]),im,ids,device='cpu');assert b['global_index'].item()==9

def test_first_applicable_classification():
 assert classify(LIMIT+1,LIMIT+1)==LABELS[0];assert classify(LIMIT,LIMIT+1)==LABELS[1];assert classify(LIMIT,LIMIT)==LABELS[2]

class SmallEncoder(torch.nn.Module):
 def encode_image(self,x):
  assert not self.training and not torch.is_grad_enabled()
  return torch.cat([x.mean((2,3)),x.square().mean((2,3))],dim=1)

def test_reference_replacement_and_absence(tmp_path):
 torch.manual_seed(7);encoder=FrozenCLIP(SmallEncoder(),torch.zeros(3,6)).eval();bank=tmp_path/'bank_images.pt';torch.save(torch.rand(3,1,3,12,16),bank);ref=tmp_path/'clean_reference.pt';target=tmp_path/'targets.json';results=[]
 for action in range(3):
  if action<2:torch.save(torch.full((1,3,12,16),float(action*900)),ref);target.write_text(str(action*99999))
  else:ref.unlink();target.unlink()
  e=extract_bank(encoder,bank,'cpu');q=context(e,torch.tensor([0,1,2]),torch.zeros(3,dtype=torch.long));nn=nearest(q['z'],q['z'],torch.tensor([0,1,2]),torch.tensor([0,1,2]),torch.arange(3),loo=True,device='cpu');results.append((e,q['z'],nn))
 for e,z,nn in results[1:]:
  assert torch.equal(e,results[0][0]) and torch.equal(z,results[0][1]);assert all(torch.equal(v,results[0][2][k]) for k,v in nn.items())
