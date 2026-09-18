import torch,hashlib
from torch import nn
from ttie.energy_model import EnergyHead
def thash(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def relative_value(prediction,anchor_prediction,target,anchor_target):
 return nn.functional.huber_loss(prediction-anchor_prediction,target-anchor_target,delta=1.)
def train_scalar(features,mse,anchors):
 torch.manual_seed(7);torch.set_num_threads(1)
 features=features.detach().cpu().float();target=(mse.detach().cpu().double()+1e-6).log();head=EnergyHead();initial={k:thash(v) for k,v in head.state_dict().items()}
 with torch.no_grad():
  head.x_mean.copy_(features.double().mean(0));scale=features.double().std(0,unbiased=False);head.x_scale.copy_(torch.where(scale==0,torch.ones_like(scale),scale));head.y_mean.copy_(target.mean());scale=target.std(unbiased=False);head.y_scale.copy_(torch.where(scale==0,torch.ones_like(scale),scale))
 target=(target.float()-head.y_mean)/head.y_scale;optimizer=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=1e-4);generator=torch.Generator().manual_seed(7);history=[]
 for epoch in range(100):
  order=torch.randperm(len(features),generator=generator);total=0.
  for batch in order.split(256):
   x=features[batch];prediction=head.standardized(x);value_loss=relative_value(prediction,head.standardized(features[anchors[batch]]),target[batch],target[anchors[batch]])
   optimizer.zero_grad(set_to_none=True);value_loss.backward();optimizer.step();total+=float(value_loss.detach())*len(batch)
  history.append(dict(epoch=epoch+1,train_huber=total/len(features),order_sha256=thash(order)))
 head.zero_grad(set_to_none=True);head.eval().requires_grad_(False)
 return head,history,initial,optimizer.state_dict(),generator.get_state()
