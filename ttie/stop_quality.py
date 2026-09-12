"""Fixed source-trained T012 quality regressor; inference needs only 33 features."""
import torch
from torch import nn
from .stop_trajectory import SCHEMA

RECIPE=dict(input_dim=33,hidden=[64,64],activation='ReLU',loss='Huber',delta=1.,
            optimizer='AdamW',lr=1e-3,weight_decay=1e-4,betas=[.9,.999],eps=1e-8,
            batch_size=256,epochs=100,seed=7,checkpoint='final_epoch',device='cpu',
            normalization='train rows only, population std; constant scales=1',target='log(MSE+1e-6)')


class QualityHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(33,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1))
        self.register_buffer('x_mean',torch.zeros(33));self.register_buffer('x_scale',torch.ones(33))
        self.register_buffer('y_mean',torch.zeros(()));self.register_buffer('y_scale',torch.ones(()))

    def standardized(self,features):
        features=features.to(self.x_mean)
        return self.net((features-self.x_mean)/self.x_scale).flatten()

    def forward(self,features):return self.standardized(features)*self.y_scale+self.y_mean

    def normalization(self):return {k:getattr(self,k).cpu().tolist() for k in ('x_mean','x_scale','y_mean','y_scale')}


def train_head(features,mse):
    # No calibration data, clean pixels, condition or image IDs are inputs here.
    torch.manual_seed(7);torch.set_num_threads(1)
    features=features.detach().cpu().float();target=(mse.detach().cpu().double()+1e-6).log()
    head=QualityHead()
    with torch.no_grad():
        head.x_mean.copy_(features.double().mean(0));scale=features.double().std(0,unbiased=False)
        head.x_scale.copy_(torch.where(scale==0,torch.ones_like(scale),scale))
        head.y_mean.copy_(target.mean());scale=target.std(unbiased=False)
        head.y_scale.copy_(torch.where(scale==0,torch.ones_like(scale),scale))
    target=(target.float()-head.y_mean)/head.y_scale
    optimizer=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=1e-4)
    generator=torch.Generator().manual_seed(7);history=[]
    for epoch in range(100):
        order=torch.randperm(len(features),generator=generator);total=0.
        for batch in order.split(256):
            prediction=head.standardized(features[batch]);loss=nn.functional.huber_loss(prediction,target[batch],delta=1.)
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            total+=float(loss.detach())*len(batch)
        history.append(dict(epoch=epoch+1,train_huber=total/len(features)))
    head.zero_grad(set_to_none=True);head.eval().requires_grad_(False)
    return head,history


def save_head(head,path):torch.save(dict(state_dict=head.state_dict(),schema=SCHEMA,recipe=RECIPE),path)


def load_head(path):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    assert saved['schema']==SCHEMA and saved['recipe']==RECIPE
    head=QualityHead();head.load_state_dict(saved['state_dict']);return head.eval().requires_grad_(False)
