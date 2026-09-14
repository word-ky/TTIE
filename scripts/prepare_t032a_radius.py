"""Freeze source-only cross-image support radius before fresh inference."""
import argparse,json,hashlib,time,math
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
from scipy.spatial.distance import cdist
from PIL import Image
from ttie.support_trust import cross_image_distances

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')

def main():
    p=argparse.ArgumentParser()
    for k in ['bank-root','receipt','checkpoint','out']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    start=time.perf_counter()
    def deny(*args,**kwargs):raise AssertionError('source support must not decode images')
    Image.open=deny
    receipt=json.loads(a.receipt.read_bytes());manifest=json.loads((a.bank_root/'training_manifest.json').read_bytes())
    assert sha(a.bank_root/'training_manifest.json')==receipt['source_records_manifest_sha256']
    assert manifest==receipt['source_records_manifest']
    assert sha(a.checkpoint)==receipt['energy_sha256']=='c3d1eef9f20af163e1268823cf16d3fdaed315e7723fc94db3b1138ad1336521'
    ids=sorted(r['image_id'] for r in receipt['split_manifests']['train_t014_sobolev']);assert len(set(ids))==80
    chunks=[];row_ids=[];bindings=[]
    for r in manifest:
        assert r['image_id'] in ids
        path=a.bank_root/r['directory']/'bank.pt';assert sha(path)==r['files']['bank.pt']['sha256']
        x=torch.load(path,map_location='cpu',weights_only=True)['features'];assert len(x)==r['states']
        chunks.append(x);row_ids.extend([ids.index(r['image_id'])]*len(x))
        bindings.append(dict(image_id=r['image_id'],directory=r['directory'],rows=len(x),sha256=sha(path)))
    source=torch.cat(chunks);assert source.shape==(7346,28)
    head=torch.load(a.checkpoint,map_location='cpu',weights_only=True)['state_dict'];mean=head['x_mean'];scale=head['x_scale']
    assert torch.equal(source.double().mean(0).float(),mean)
    std=source.double().std(0,unbiased=False);assert torch.equal(torch.where(std==0,torch.ones_like(std),std).float(),scale)
    row_ids=np.asarray(row_ids);z=((source.double()-mean.double())/scale.double()).numpy();assert np.isfinite(z).all()
    torch.cuda.synchronize();gpu_start=time.perf_counter()
    values=cross_image_distances(source.cuda(),torch.as_tensor(row_ids,device='cuda'),mean.cuda(),scale.cuda()).cpu().numpy()
    torch.cuda.synchronize();gpu_seconds=time.perf_counter()-gpu_start
    independent=[]
    for i in range(0,len(z),128):
        d=cdist(z[i:i+128],z);d[row_ids[i:i+len(d),None]==row_ids[None,:]]=np.inf
        independent.extend(d.min(1)/math.sqrt(28))
    error=float(np.max(np.abs(values-independent)));assert error<=1e-9 and np.isfinite(values).all()
    radius=float(np.quantile(values,.95,method='linear'))
    replay_radius=float(np.quantile(independent,.95,method='linear'));assert abs(radius-replay_radius)<=1e-9
    a.out.mkdir(parents=True,exist_ok=False)
    write(a.out/'cross_distances.json',[dict(row=i,image_id=ids[int(row_ids[i])],distance=float(v)) for i,v in enumerate(values)])
    torch.save(dict(source_features=source,row_ids=torch.as_tensor(row_ids),x_mean=mean,x_scale=scale),a.out/'source.pt')
    write(a.out/'freeze.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),source_rows=7346,source_ids=ids,source_bank_files=bindings,
        source_manifest_sha256=sha(a.bank_root/'training_manifest.json'),t014_receipt_sha256=sha(a.receipt),checkpoint_sha256=sha(a.checkpoint),
        source_sha256=sha(a.out/'source.pt'),cross_distances_sha256=sha(a.out/'cross_distances.json'),r_support=radius,percentile=.95,quantile_method='linear',
        independent_distance_max_abs_error=error,independent_radius_abs_error=abs(radius-replay_radius),image_decodes=0,normal_decodes=0,
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,gpu_seconds=gpu_seconds,total_seconds=time.perf_counter()-start))
    write(a.out/'spec.json',dict(radius_freeze_sha256=sha(a.out/'freeze.json'),r_support=radius,percentile=.95,quantile_method='linear',
        rule='first d_NN > radius; step0 exit selects identity; otherwise prefix before exit; minimum energy earliest tie'))
    print('RADIUS FREEZE PASS',radius,'independent max error',error,flush=True)

if __name__=='__main__':main()
