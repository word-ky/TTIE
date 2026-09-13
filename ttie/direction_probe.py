"""T018-C fixed direction classifier, adapted from the accepted CPU head lifecycle."""
import torch
from torch import nn

CLASSES=(.5,.4,.6)
RECIPE=dict(input_dim=84,hidden=[64,64],output_dim=3,activation='SiLU',loss='unweighted_cross_entropy',
    optimizer='AdamW',lr=1e-3,weight_decay=1e-4,betas=[.9,.999],optimizer_eps=1e-8,
    batch_size=256,epochs=100,seed=7,checkpoint='final_epoch',device='cpu',
    normalization='training rows only; population std clamped at 1e-12',normalization_epsilon=1e-12,
    class_order=list(CLASSES),training_order='seed7 randperm per epoch; original training row order')


def axis_features(saved):
    f=torch.as_tensor(saved,dtype=torch.float32);assert f.ndim==3 and f.shape[1:]==(9,28)
    center=f[:,4]
    return torch.stack([torch.cat((center,f[:,low]-center,f[:,high]-center),dim=1) for low,high in ((1,7),(3,5))],dim=1)


class DirectionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(84,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,3))
        self.register_buffer('x_mean',torch.zeros(84));self.register_buffer('x_scale',torch.ones(84))

    def forward(self,z):return self.net((z.to(self.x_mean)-self.x_mean)/self.x_scale)


def fit_head(train_z,train_classes):
    torch.manual_seed(7);torch.set_num_threads(1)
    z=train_z.detach().cpu().float();target=train_classes.detach().cpu().long();head=DirectionHead()
    with torch.no_grad():
        head.x_mean.copy_(z.double().mean(0));head.x_scale.copy_(z.double().std(0,unbiased=False).clamp_min(1e-12))
    optimizer=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=1e-4)
    generator=torch.Generator().manual_seed(7);history=[]
    for epoch in range(100):
        order=torch.randperm(len(z),generator=generator);total=0.
        for batch in order.split(256):
            loss=nn.functional.cross_entropy(head(z[batch]),target[batch])
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();total+=float(loss.detach())*len(batch)
        history.append(dict(epoch=epoch+1,train_cross_entropy=total/len(z)))
    head.zero_grad(set_to_none=True);head.eval().requires_grad_(False)
    return head,history


@torch.no_grad()
def predict(head,heldout_z):
    logits=head(heldout_z);return logits.tolist(),logits.argmax(1).tolist()


def fit_axis_fold(z,targets,fold,axis):
    # This is the only target lookup. No held-out target or metadata is passed to the head.
    labels=torch.tensor([CLASSES.index(targets[i]['bx' if axis==0 else 'by']) for i in fold['train']])
    head,history=fit_head(z[fold['train'],axis],labels)
    logits,classes=predict(head,z[fold['heldout'],axis])
    return head,history,logits,classes
