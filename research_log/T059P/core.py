import torch
LIMIT=.07650849781930447
LABELS=['frozen CLIP-latent locality fails the source control; stop','full frozen CLIP latent context is still insufficient for source-image scalar localization under fixed 1-NN','frozen CLIP latent context restores source-selector scalar locality; prompt-score compression is supported as a bottleneck candidate for a later parametric test']
def classify(f,s):return LABELS[0] if f>LIMIT else LABELS[1] if s>LIMIT else LABELS[2]
def extract_bank(encoder,path,device):
 images=torch.load(path,weights_only=True,map_location='cpu');assert images.ndim==5 and images.shape[1]==1
 with torch.no_grad():return torch.stack([encoder.image_embeddings(image.to(device)).cpu() for image in images])
def context(e,bank,state):
 a=torch.empty(len(e),dtype=torch.long)
 for b in bank.unique():
  q=torch.where((bank==b)&(state==0))[0];assert len(q)==1;a[bank==b]=q[0]
 e0=e[a];return dict(anchor=a,e=e,e0=e0,z=torch.cat(((e-e0).flatten(1),e0.flatten(1)),dim=1))
def nearest(query,train,query_images,train_images,global_train,loo=False,device='cuda:0'):
 # Canonical ascending candidate order makes torch.min choose smallest global ID on ties.
 assert torch.equal(global_train,global_train.sort().values)
 tr=train.to(device,dtype=torch.float64);ti=train_images.to(device);neighbors=[];distances=[];ties=[]
 for start in range(0,len(query),4):
  q=query[start:start+4].to(device,dtype=torch.float64)
  d=(q[:,None,:]-tr[None,:,:]).square().sum(-1)
  if loo:d.masked_fill_(query_images[start:start+4].to(device)[:,None]==ti[None,:],float('inf'))
  dist,idx=d.min(1);assert torch.isfinite(dist).all()
  neighbors.append(idx.cpu());distances.append(dist.cpu());ties.append((d==dist[:,None]).sum(1).cpu())
 return dict(index=torch.cat(neighbors),global_index=global_train[torch.cat(neighbors)],distance=torch.cat(distances),tie_count=torch.cat(ties))
