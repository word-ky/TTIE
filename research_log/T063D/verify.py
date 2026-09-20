"""Separate 300-output hash and independent metric/aggregate/access replay."""
import argparse,json,zipfile,io,hashlib,math,statistics
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
from PIL import Image
from scripts.evaluate_t026a import independent_ssim

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(out):
    from research_log.T063D.verify_states import verify as verify_states
    state_verification=verify_states(out)
    torch.set_num_threads(1);manifest=Path('research_log/T063D/manifest.json');assert digest(manifest)=='3206ea57061f4b45164a81f105f818de6d6d15b342a77797ce9f1eaaccc52554'
    rows=json.loads(manifest.read_bytes())['selected'];freeze=json.loads((out/'freeze.json').read_bytes());paired=json.loads((out/'paired.json').read_bytes());summary=json.loads((out/'result.json').read_bytes());opened=json.loads((out/'reference_reads.json').read_bytes());marker=json.loads((out/'reference_open.json').read_bytes())
    assert digest(out/'freeze.json')==marker['freeze_sha256']==summary['freeze_sha256']
    assert freeze['completed_utc']<marker['first_reference_read_utc']<=min(r['utc'] for r in opened)
    assert digest(out/'config.json')==freeze['config_sha256'] and len(rows)==len(paired)==len(opened)==100 and freeze['outputs']==300
    errors=[];allmetrics=[]
    with zipfile.ZipFile('/media/wenchang/F/wjq/TTIE/shared/t022a/LOL-v2.zip') as z:
        for i,(r,f,p,o) in enumerate(zip(rows,freeze['rows'],paired,opened)):
            assert r['low']==f['low']==p['low'] and r['normal']==p['normal']==o['normal'];directory=out/f'{i:03d}'
            assert digest(directory/'outputs.pt')==f['outputs_sha256'];saved=torch.load(directory/'outputs.pt',weights_only=True,map_location='cpu')
            for n,t in saved.items():
                h=hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest();assert h==(f['identity_hash'] if n=='identity' else f['methods'][n]['output_hash'])
                if n!='identity':assert digest(directory/(n+'_trace.pt'))==f['methods'][n]['trace_sha256']
            t=torch.load(directory/'T063_trace.pt',weights_only=True,map_location='cpu');assert len(t['gradients'])==27 and len(t['states'])==28 and t['selected_step']==f['methods']['T063']['selected_step'] and torch.count_nonzero(t['states'][0])==0
            raw=z.read('LOL-v2/Real_captured/'+r['normal']);assert hashlib.sha256(raw).hexdigest()==o['sha256']
            with Image.open(io.BytesIO(raw)) as im:target=np.asarray(im.convert('RGB'),dtype=np.float64)/255
            target=target.astype('float32').astype('float64');metrics={}
            for name,t in saved.items():
                x=t.numpy()[0].transpose(1,2,0).astype('float64');d=(x-target).reshape(-1);psnr=-10*math.log10(float(d@d)/len(d));ssim=independent_ssim(x,target)
                errors.extend([abs(psnr-p['metrics'][name]['psnr']),abs(ssim-p['metrics'][name]['ssim'])]);metrics[name]=dict(psnr=psnr,ssim=ssim)
            allmetrics.append(metrics)
    assert max(errors)<1e-10
    d=[r['T063']['psnr']-r['T036']['psnr'] for r in allmetrics];b=[r['T063']['psnr']-r['T026']['psnr'] for r in allmetrics];s=[r['T063']['ssim']-r['T036']['ssim'] for r in allmetrics]
    vals=dict(mean_delta_psnr=math.fsum(d)/100,median_delta_psnr=statistics.median(d),regressions_t026=sum(v<0 for v in b),worst_delta_t026=min(b),mean_delta_ssim=math.fsum(s)/100)
    for k,v in vals.items():assert abs(v-summary[k])<1e-10
    for n in ['T026','T036','T063','identity']:
        for k in ['psnr','ssim']:assert abs(math.fsum(r[n][k] for r in allmetrics)/100-summary['absolute'][n][k])<1e-10
    gates=dict(mean_psnr=vals['mean_delta_psnr']>=2,median_psnr=vals['median_delta_psnr']>0,regressions=vals['regressions_t026']<=29,worst=vals['worst_delta_t026']>=-5.614,mean_ssim=vals['mean_delta_ssim']>=-.001)
    assert gates==summary['gates'];verdict='FRESH_QUALIFICATION_PASS' if all(gates.values()) else 'FRESH_QUALIFICATION_NEGATIVE';assert verdict==summary['verdict']
    v=dict(state_verification=state_verification,status='PASS',verdict=verdict,final_outputs=300,metrics_checked=800,max_metric_error=max(errors),gates=gates,cohort_sha256=digest(manifest),verified_utc=datetime.now(timezone.utc).isoformat())
    with (out/'verification.json').open('x') as f:json.dump(v,f,indent=2)
    print(json.dumps(v),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
