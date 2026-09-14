"""Bind source-only features and freeze all support distances before reference access."""
import argparse,json,hashlib,time,math
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
from PIL import Image
from scipy.spatial.distance import cdist
from core import distances
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')
def main():
    p=argparse.ArgumentParser()
    for k in ['bank-root','receipt','accepted','manifest','checkpoint','out']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(7);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;start=time.perf_counter();opened=[]
    def deny(*args,**kwargs):opened.append(str(args[0]));raise AssertionError('support stage must not decode any image')
    Image.open=deny
    receipt=json.loads(a.receipt.read_bytes());bank_manifest=json.loads((a.bank_root/'training_manifest.json').read_bytes())
    assert sha(a.bank_root/'training_manifest.json')==receipt['source_records_manifest_sha256']
    assert bank_manifest==receipt['source_records_manifest']
    assert sha(a.checkpoint)==receipt['energy_sha256']=='c3d1eef9f20af163e1268823cf16d3fdaed315e7723fc94db3b1138ad1336521'
    ids={r['image_id'] for r in receipt['split_manifests']['train_t014_sobolev']};assert len(ids)==80
    chunks=[];bindings=[]
    for r in bank_manifest:
        assert r['image_id'] in ids
        path=a.bank_root/r['directory']/'bank.pt';assert sha(path)==r['files']['bank.pt']['sha256']
        x=torch.load(path,map_location='cpu',weights_only=True)['features'];assert len(x)==r['states']
        chunks.append(x);bindings.append(dict(image_id=r['image_id'],directory=r['directory'],rows=len(x),sha256=sha(path)))
    source=torch.cat(chunks);assert source.shape==(7346,28) and {r['image_id'] for r in bank_manifest}==ids
    head=torch.load(a.checkpoint,map_location='cpu',weights_only=True)['state_dict'];mean=head['x_mean'];scale=head['x_scale']
    assert torch.equal(source.double().mean(0).float(),mean)
    std=source.double().std(0,unbiased=False);assert torch.equal(torch.where(std==0,torch.ones_like(std),std).float(),scale)
    cohort=json.loads(a.manifest.read_bytes());freeze=json.loads((a.accepted/'freeze.json').read_bytes())
    assert sha(a.manifest)==freeze['manifest_sha256']=='ec67f0a6af5682c8e1e929db56e1d771dfd3183f75cb4b365052cde024f55f2d'
    states=[];raws=[];rows=[];accepted_bindings=[]
    for r,s in zip(freeze['rows'],cohort['selected']):
        assert r['low']==s['low'];d=a.accepted/f'{r["index"]:03d}'
        for name,h in r['files'].items():assert sha(d/name)==h['sha256']
        t=torch.load(d/'trajectory.pt',map_location='cpu',weights_only=True)
        assert t['features'].shape==(41,28);states.append(t['features']);raws.append(t['states'])
        decision=json.loads((d/'decision.json').read_bytes())
        accepted_bindings.append(dict(index=r['index'],low=r['low'],files=r['files'],gate=decision['gate'],original_step=decision['original']['selected_step']))
        rows.extend(dict(index=r['index'],image=r['low'],step=j,original_selected=j==decision['original']['selected_step']) for j in range(41))
    state_features=torch.cat(states);assert state_features.shape==(4100,28)
    z=(source.double()-mean.double())/scale.double();q=(state_features.double()-mean.double())/scale.double()
    assert torch.isfinite(z).all() and torch.isfinite(q).all()
    torch.cuda.synchronize();gpu_start=time.perf_counter()
    values=distances(source.cuda(),state_features.cuda(),mean.cuda(),scale.cuda()).cpu().numpy();torch.cuda.synchronize()
    gpu_seconds=time.perf_counter()-gpu_start
    independent=np.concatenate([cdist(chunk,z.numpy()).min(1)/math.sqrt(28) for chunk in np.array_split(q.numpy(),33)])
    error=float(np.max(np.abs(values-independent)));assert error<=1e-9 and np.isfinite(values).all()
    a.out.mkdir(parents=True,exist_ok=False)
    for r,v in zip(rows,values):r['distance']=float(v)
    write(a.out/'scores.json',rows)
    torch.save(dict(source_features=source,state_features=state_features,raw_states=torch.stack(raws),x_mean=mean,x_scale=scale),a.out/'bound_features.pt')
    assert not opened
    write(a.out/'freeze.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),source_rows=7346,source_ids=sorted(ids),source_bank_files=bindings,
        source_manifest_sha256=sha(a.bank_root/'training_manifest.json'),t014_receipt_sha256=sha(a.receipt),checkpoint_sha256=sha(a.checkpoint),
        accepted=str(a.accepted),accepted_freeze_sha256=sha(a.accepted/'freeze.json'),cohort_sha256=sha(a.manifest),accepted_bindings=accepted_bindings,
        scores_sha256=sha(a.out/'scores.json'),bound_features_sha256=sha(a.out/'bound_features.pt'),states=4100,
        image_decodes=0,normal_decodes=0,independent_distance_max_abs_error=error,distance_dtype='float64',
        normalization='saved float32 features/buffers promoted tofloat64 for fixed formula; head unchanged',
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,gpu_distance_seconds=gpu_seconds,total_seconds=time.perf_counter()-start))
    print('SUPPORT FREEZE PASS4100 source7346; independent maxerror',error,flush=True)
if __name__=='__main__':main()
