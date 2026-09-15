"""Bind all200 accepted frozen outputs before any normal, then score unchanged tensors."""
from core import *
import argparse,csv,time,math,torch
from PIL import Image
from ttie.ssim_transfer import rgb_ssim,METRIC
p=argparse.ArgumentParser()
for k in ['baseline','candidate','manifest','normal-root','binding','out']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();start=time.perf_counter();torch.set_num_threads(1)
assert sha(a.manifest)==COHORT
cohort=json.loads(a.manifest.read_bytes())['selected'];assert len(cohort)==100
code=json.loads(a.binding.read_bytes())
for name,h in code.items():assert sha(name)==h
roots=dict(baseline=a.baseline,candidate=a.candidate);freezes={}
for name,path in roots.items():
    assert sha(path/'freeze.json')==FREEZES[name]
    freezes[name]=json.loads((path/'freeze.json').read_bytes());assert freezes[name]['cohort_sha256']==COHORT and len(freezes[name]['rows'])==100
assert freezes['baseline']['assets']==freezes['candidate']['assets']
original=Image.open
def denied(*args,**kwargs):raise AssertionError('Normal/image open before all output identities bound')
Image.open=denied;a.out.mkdir(parents=True,exist_ok=False);bound=[];cache=[]
for i,item in enumerate(cohort):
    pair={};images={};saved_pair={}
    for name,root in roots.items():
        rec=freezes[name]['rows'][i];assert rec['index']==i and rec['low']==item['low']
        file=root/rec['file'];assert sha(file)==rec['sha256'];saved=torch.load(file,map_location='cpu',weights_only=True)
        idx=saved['gains'].index(1.75);assert saved['gains']==([1.25,1.75] if name=='baseline' else [1.75])
        raw=saved['states'][idx];image=saved['outputs'][idx];active=torch.tensor(saved['gate']['active'],dtype=torch.bool)
        expected=torch.full((4,),math.atanh(math.log(1.75)/math.log(2)),dtype=raw.dtype);expected[~active]=0
        assert torch.equal(raw[0,2].flatten(),expected) and torch.isfinite(image).all() and image.shape==(1,3,400,600)
        assert thash(image)==saved['output_hashes'][idx]==rec['output_hashes'][idx]
        if name=='candidate':assert saved['legacy_step']==rec['legacy_step']==10 and thash(raw[:,:2])==rec['legacy_sha256'] and torch.equal(raw[:,:2],saved['legacy'])
        else:assert saved['selected_step']==rec['selected_step']
        pair[name]=dict(path=str(file),sha256=rec['sha256'],output_index=idx,output_sha256=thash(image),raw_sha256=thash(raw))
        images[name]=image.clone();saved_pair[name]=saved
    assert saved_pair['baseline']['gate']==saved_pair['candidate']['gate']
    bound.append(dict(index=i,low=item['low'],normal=item['normal'],normal_sha256=item['normal_sha256'],**pair));cache.append(images)
write(a.out/'output_binding.json',dict(label=LABEL,completed_utc=utc(),pairs=bound,cohort_sha256=COHORT,freeze_sha256=FREEZES,normal_opens=0,optimizer_updates=0,state_changes=0,selection_changes=0,code_binding=code,metric=METRIC,psnr='float64 per-image RGB MSE over float32 native pixels; -10log10; no crop/resize',rerendered=False))
opened=[];allowed={str((a.normal_root/r['normal']).resolve()) for r in cohort}
def normal_open(path,*args,**kwargs):
    name=str(Path(path).resolve());assert name in allowed;opened.append(dict(path=name,utc=utc()));return original(path,*args,**kwargs)
Image.open=normal_open;rows=[]
for item,images,rec in zip(cohort,cache,bound):
    path=a.normal_root/item['normal'];assert sha(path)==item['normal_sha256']
    with Image.open(path) as im:normal=(np.asarray(im.convert('RGB'),dtype=np.float64)/255.).astype(np.float32).astype(np.float64)
    row=dict(index=rec['index'],low=item['low'],normal=item['normal'])
    for name,image in images.items():
        x=image.squeeze(0).permute(1,2,0).numpy().astype(np.float64);assert x.shape==normal.shape
        row[name+'_psnr']=float(-10*np.log10(np.mean((x-normal)**2)));row[name+'_ssim']=rgb_ssim(x,normal)
        assert thash(image)==rec[name]['output_sha256']
    for metric in ['psnr','ssim']:row['delta_'+metric]=row['candidate_'+metric]-row['baseline_'+metric]
    assert all(np.isfinite(v) for k,v in row.items() if k not in ['low','normal']);rows.append(row)
assert len(rows)==len(opened)==100
with (a.out/'metrics.csv').open('w',newline='') as stream:
    w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=aggregate(rows);summary['label']=LABEL;write(a.out/'summary.json',summary)
for rec in bound:
    for name in roots:assert sha(rec[name]['path'])==rec[name]['sha256']
write(a.out/'receipt.json',dict(completed_utc=utc(),output_binding_sha256=sha(a.out/'output_binding.json'),opened_normals=opened,pairs=100,output_hashes_unchanged=True,optimizer_updates=0,state_changes=0,selection_changes=0,official_test_access=False,rerendered=False,seconds=time.perf_counter()-start,device='CPU: unchanged accepted SciPy float64 metric',files={n:sha(a.out/n) for n in ['metrics.csv','summary.json']}))
print(json.dumps(summary),flush=True)
