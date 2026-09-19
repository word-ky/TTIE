"""Offline scoring of the immutable T062-A states; no inference or rendering."""
import argparse,csv,hashlib,json,os,time,statistics
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from ttie.ssim_transfer import rgb_ssim
from scripts.evaluate_t026a import pixels,independent_ssim

RAW=Path('/media/wenchang/F/wjq/TTIE/runs/T062A-fixed-zr')
ROOT=Path('/home/wenchang/asdasdsad/wjq/TTIE')
FREEZE='301c041aedeb7d4925560a860417891e6bc64fa616da3c1c10565b10d3496d00'
COHORT='279c74b335999d6631d436c79ea80e9e2cccfc4b97cb7c0b992829d04cfaf40b'
BASE='5676541d245fdb41c54a543cf88a79ca37640eea94432af0665cca2ddce2277c'
def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,d):
    with p.open('x') as f:json.dump(d,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())

def summary(rows,k):
    p=[r['psnr'][k]-r['t036_psnr'] for r in rows];b=[r['psnr'][k]-r['t026_psnr'] for r in rows]
    s=[r['ssim'][k]-r['t036_ssim'] for r in rows]
    gates=dict(regressions=sum(x<0 for x in b)<=29,worst=min(b)>=-5.614,ssim=statistics.fmean(s)>=-.001)
    return dict(step=k,mean_psnr=statistics.fmean(r['psnr'][k] for r in rows),mean_ssim=statistics.fmean(r['ssim'][k] for r in rows),
        mean_delta=statistics.fmean(p),median_delta=statistics.median(p),improve=sum(x>0 for x in p),regress=sum(x<0 for x in p),tie=sum(x==0 for x in p),
        regressions_t026=sum(x<0 for x in b),worst_delta_t026=min(b),mean_delta_ssim=statistics.fmean(s),safety_gates=gates,safety_eligible=all(gates.values()))

def select(table):
    eligible=[r for r in table if r['safety_eligible']]
    chosen=max(eligible,key=lambda r:(r['mean_delta'],-r['step'])) if eligible else None
    gates=dict(safety=chosen is not None,mean_psnr=chosen is not None and chosen['mean_delta']>=2.,median_psnr=chosen is not None and chosen['median_delta']>0)
    return dict(k_star=None if chosen is None else chosen['step'],chosen=chosen,eligible_steps=[r['step'] for r in eligible],gates=gates,
        verdict='PASS' if all(gates.values()) else 'NEGATIVE',classification='the T062 zero-reference trajectory is useful and a global early stop rescues its safety/structure failure' if all(gates.values()) else 'the global-early-stop rescue is insufficient')

def score(args):
    rec,item,base,old=args;torch.set_num_threads(1)
    directory=RAW/f"{rec['index']:03d}"
    for n,h in rec['files'].items():assert sha(directory/n)==h
    images=torch.load(directory/'images.pt',map_location='cpu',weights_only=True)
    assert images.shape[0]==41 and torch.isfinite(images).all()
    target=ROOT/'shared/t036a/normal'/item['normal'];assert sha(target)==item['normal_sha256']
    normal=pixels(target).astype(np.float32).astype(np.float64)
    psnr=[];ssim=[];errors=[];state_hashes=[]
    for k,im in enumerate(images):
        h=hashlib.sha256(im.contiguous().numpy().tobytes()).hexdigest();assert h==rec['rendered_hashes'][k];state_hashes.append(h)
        x=im[0].permute(1,2,0).numpy().astype(np.float64);d=x-normal
        p=float(-10*np.log10(np.mean(d*d)));s=rgb_ssim(x,normal)
        independent_p=float(-10*np.log10(np.dot(d.ravel(),d.ravel())/d.size));independent_s=independent_ssim(x,normal)
        error=max(abs(p-independent_p),abs(s-independent_s));assert error<1e-11
        psnr.append(p);ssim.append(s);errors.append(error)
    k=rec['selected_step'];assert abs(psnr[k]-old['psnr'])<1e-10 and abs(ssim[k]-old['ssim'])<1e-10
    return dict(index=rec['index'],low=item['low'],psnr=psnr,ssim=ssim,t036_psnr=float(base['common_selected_psnr']),t036_ssim=float(base['common_selected_ssim']),t026_psnr=float(base['baseline_selected_psnr']),
        state_hashes=state_hashes,normal_sha256=item['normal_sha256'],independent_metric_errors=errors)

def main(out):
    torch.set_num_threads(1);start=time.perf_counter();assert sha(RAW/'freeze.json')==FREEZE
    assert sha(RAW/'paired.json')=='dfcf67ec1d427d1e2bdf775d3a0ad5eeb0b9c8d81bf735f0cd8bd8428cf54273'
    freeze=json.loads((RAW/'freeze.json').read_bytes());assert len(freeze['rows'])==100 and sha(RAW/'config.json')==freeze['config_sha256']
    manifest=Path('research_log/T036A_cohort/manifest.json');base=Path('research_log/T037A_result/per_image.csv')
    assert sha(manifest)==COHORT and sha(base)==BASE
    binding=json.loads(Path('research_log/T062B/binding.json').read_bytes())
    for n,h in binding.items():assert sha(n)==h,n
    items=json.loads(manifest.read_bytes())['selected'];bases=list(csv.DictReader(base.open()));previous=json.loads((RAW/'paired.json').read_bytes())
    assert len(items)==len(bases)==len(previous)==100
    jobs=[]
    for i,(rec,item,b,old) in enumerate(zip(freeze['rows'],items,bases,previous)):
        assert rec['index']==old['index']==int(b['index'])==i and rec['low']==item['low']==b['low']==old['low']
        jobs.append((rec,item,b,old))
    out.mkdir(parents=True,exist_ok=False)
    write(out/'inputs.json',dict(freeze_sha256=FREEZE,config_sha256=freeze['config_sha256'],cohort_sha256=COHORT,baseline_sha256=BASE,previous_paired_sha256=sha(RAW/'paired.json'),source_binding=binding,started_utc=utc(),workers=8,mode='offline development hyperparameter selection only'))
    rows=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for row in pool.map(score,jobs):
            rows.append(row);write(out/f"{row['index']:03d}.json",row);print('scored',len(rows),flush=True)
    table=[summary(rows,k) for k in range(41)];selection=select(table)
    write(out/'table.json',table);write(out/'selection.json',dict(**selection,completed_utc=utc(),seconds=time.perf_counter()-start,table_sha256=sha(out/'table.json'),new_optimizer_runs=0,new_render_runs=0,official_test_access=0,cross_dataset_access=0))
    print(json.dumps(selection),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
