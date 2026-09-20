import argparse,json,io,time,zipfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from ttie.ssim_transfer import rgb_ssim
from research_log.T063D.infer import HERE,MANIFEST,COHORT,ARCHIVE,sha,utc,write,tensor_hash

def main(out):
    torch.set_num_threads(1);begin=time.perf_counter();f=json.loads((out/'freeze.json').read_bytes())
    assert sha(MANIFEST)==COHORT==f['cohort_sha256'] and f['outputs']==300 and len(f['rows'])==100
    assert sha(out/'config.json')==f['config_sha256']
    for r in f['rows']:
        d=out/f"{r['index']:03d}";assert sha(d/'outputs.pt')==r['outputs_sha256']
        for n,m in r['methods'].items():assert sha(d/(n+'_trace.pt'))==m['trace_sha256']
    first=utc()
    assert first>f['completed_utc'];write(out/'reference_open.json',dict(first_reference_read_utc=first,freeze_sha256=sha(out/'freeze.json')))
    assert sha(HERE/'evaluation_binding.json')==json.loads((HERE/'method.json').read_bytes())['evaluation_binding_sha256']
    evaluation_binding=json.loads((HERE/'evaluation_binding.json').read_bytes())
    for n,h in evaluation_binding.items():assert sha(n)==h,n
    rows=json.loads(MANIFEST.read_bytes())['selected'];refs=json.loads((HERE/'reference_inputs.json').read_bytes())
    result=[];opens=[]
    with zipfile.ZipFile(ARCHIVE) as z:
        for i,(r,ref,fr) in enumerate(zip(rows,refs,f['rows'])):
            assert r['normal']==ref['normal'] and r['low']==fr['low'];stamp=utc();raw=z.read('LOL-v2/Real_captured/'+r['normal']);import hashlib
            assert hashlib.sha256(raw).hexdigest()==ref['normal_sha256']
            with Image.open(io.BytesIO(raw)) as im:normal=(np.asarray(im.convert('RGB'),dtype=np.float64)/255).astype(np.float32).astype(np.float64)
            opens.append(dict(normal=r['normal'],utc=stamp,sha256=ref['normal_sha256']))
            outputs=torch.load(out/f'{i:03d}'/'outputs.pt',weights_only=True,map_location='cpu');metrics={}
            for name,t in outputs.items():
                expected=fr['identity_hash'] if name=='identity' else fr['methods'][name]['output_hash'];assert tensor_hash(t)==expected
                image=t[0].permute(1,2,0).numpy().astype(np.float64);mse=float(np.mean((image-normal)**2))
                metrics[name]=dict(psnr=float(-10*np.log10(mse)),ssim=rgb_ssim(image,normal))
            result.append(dict(index=i,low=r['low'],normal=r['normal'],metrics=metrics))
    deltas=lambda method,key:np.array([r['metrics']['T063'][key]-r['metrics'][method][key] for r in result])
    d=deltas('T036','psnr');b=deltas('T026','psnr');s=deltas('T036','ssim')
    gates=dict(mean_psnr=float(d.mean())>=2,median_psnr=float(np.median(d))>0,regressions=int((b<0).sum())<=29,worst=float(b.min())>=-5.614,mean_ssim=float(s.mean())>=-.001)
    summary=dict(task='T063-D',verdict='FRESH_QUALIFICATION_PASS' if all(gates.values()) else 'FRESH_QUALIFICATION_NEGATIVE',gates=gates,mean_delta_psnr=float(d.mean()),median_delta_psnr=float(np.median(d)),regressions_t026=int((b<0).sum()),worst_delta_t026=float(b.min()),mean_delta_ssim=float(s.mean()),improve=int((d>0).sum()),regress=int((d<0).sum()),tie=int((d==0).sum()),absolute={n:{k:float(np.mean([r['metrics'][n][k] for r in result])) for k in ['psnr','ssim']} for n in ['identity','T026','T036','T063']},freeze_utc=f['completed_utc'],first_reference_read_utc=first,freeze_sha256=sha(out/'freeze.json'),seconds=time.perf_counter()-begin,completed_utc=utc())
    summary['selected_step_histogram']={str(k):sum(r['methods']['T063']['selected_step']==k for r in f['rows']) for k in range(28)}
    write(out/'reference_reads.json',opens);write(out/'paired.json',result);write(out/'result.json',summary);print(json.dumps(summary),flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);main(a.parse_args().out)
