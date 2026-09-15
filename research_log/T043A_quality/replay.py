"""Independent frozen-tensor PSNR/SSIM and stdlib statistics, no T043 main helpers."""
import argparse,json,csv,hashlib,math,statistics,time
from pathlib import Path
import numpy as np,torch
from PIL import Image
from scipy.ndimage import convolve1d
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('normal_root',type=Path);a=p.parse_args();start=time.perf_counter();torch.set_num_threads(1)
binding=json.loads((a.root/'output_binding.json').read_bytes());receipt=json.loads((a.root/'receipt.json').read_bytes());published=json.loads((a.root/'summary.json').read_bytes());metrics=list(csv.DictReader((a.root/'metrics.csv').open()))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(a.root/'output_binding.json')==receipt['output_binding_sha256'] and binding['normal_opens']==0 and len(binding['pairs'])==100
assert all(x['utc']>binding['completed_utc'] for x in receipt['opened_normals'])
errors=[]
def check(x,y):
    err=abs(x-y);errors.append(err);assert err<=1e-10,(x,y,err)
weights=np.exp(-np.arange(-5,6,dtype=np.float64)**2/4.5);weights/=weights.sum()
def ssim(x,y):
    def smooth(v):return convolve1d(convolve1d(v,weights,axis=0,mode='reflect'),weights,axis=1,mode='reflect')
    mx,my=smooth(x),smooth(y)
    return float(np.mean(((2*mx*my+.0001)*(2*(smooth(x*y)-mx*my)+.0009))/((mx*mx+my*my+.0001)*(smooth(x*x)-mx*mx+smooth(y*y)-my*my+.0009))))
rows=[]
for rec,pub in zip(binding['pairs'],metrics):
    assert rec['index']==int(pub['index']) and rec['low']==pub['low']
    normal_path=a.normal_root/rec['normal'];assert sha(normal_path)==rec['normal_sha256']
    with Image.open(normal_path) as im:normal=(np.asarray(im.convert('RGB')).astype(np.float64)/255.).astype(np.float32).astype(np.float64)
    row=dict(index=rec['index'],low=rec['low'])
    for name in ['baseline','candidate']:
        info=rec[name];assert sha(info['path'])==info['sha256'];saved=torch.load(info['path'],map_location='cpu',weights_only=True);t=saved['outputs'][info['output_index']]
        assert hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()==info['output_sha256']
        x=t[0].permute(1,2,0).numpy().astype(np.float64)
        row[name+'_psnr']=float(-10*torch.log10(torch.from_numpy(x-normal).square().mean()));row[name+'_ssim']=ssim(x,normal)
    for metric in ['psnr','ssim']:row['delta_'+metric]=row['candidate_'+metric]-row['baseline_'+metric]
    for k,val in row.items():
        if k not in ['index','low']:check(val,float(pub[k]))
    rows.append(row)
def q(v,p):
    v=sorted(v);i=(len(v)-1)*p;j=math.floor(i);return v[j]+(v[math.ceil(i)]-v[j])*(i-j)
for k,mean in published['means'].items():check(statistics.mean(r[k] for r in rows),mean)
for metric,report in published['paired'].items():
    key='delta_'+metric;vals=[r[key] for r in rows];stats=dict(mean=statistics.mean(vals),median=statistics.median(vals),p10=q(vals,.1),p90=q(vals,.9),positive=sum(x>0 for x in vals),negative=sum(x<0 for x in vals),zero=sum(x==0 for x in vals))
    for k,val in stats.items():check(val,report[k])
    ordered=sorted(rows,key=lambda r:(r[key],r['index']))
    for name,subset in [('worst',ordered[:5]),('best',ordered[-5:][::-1])]:
        for r,pub in zip(subset,report[name]):assert r['index']==pub['index'] and r['low']==pub['low'];check(r[key],pub['delta'])
psnr=statistics.mean(r['delta_psnr'] for r in rows);ss=statistics.mean(r['delta_ssim'] for r in rows)
verdict='matched-gain early-state quality bridge supported' if psnr>=.5 and ss>=0 else 'matched-gain early-state quality bridge not supported / mixed'
assert verdict==published['classification']
result=dict(status='PASS',pairs=100,output_metric_evaluations=400,scalar_checks=len(errors),max_abs_error=max(errors),classification=verdict,independent_psnr='torch float64 reduction/log10',independent_ssim='explicit11tap separable convolve1d, same reflect convention',seconds=time.perf_counter()-start)
(a.root/'independent_replay.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
