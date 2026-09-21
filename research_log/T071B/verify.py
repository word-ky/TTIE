"""Independent saved-output verification; never rerun either network."""
import argparse,io,zipfile,hashlib,math,statistics
from pathlib import Path
import numpy as np,torch
from PIL import Image
from scripts.evaluate_t026a import independent_ssim
from research_log.T071A.core import pairs
from research_log.T071B.common import HERE,PAIRS,OURS,ARCHIVE,METHODS,read,write,sha,utc

def main(out):
    cohort=read(PAIRS);ours=read(OURS);result=read(out/'result.json');first=read(out/'reference_open.json')
    assert sha(ARCHIVE)==cohort['archive_sha256']
    assert sha(PAIRS)==result['pairs_manifest_sha256'] and sha(OURS)==result['ours_per_image_sha256']
    freezes={m:read(out/m/'freeze.json') for m in METHODS};tables={m:read(out/m/'per_image_metrics.json') for m in METHODS}
    errors=[];values={m:[] for m in METHODS}
    for m,f in freezes.items():
        cfg=read(out/m/'config.json')
        for p,h in cfg['source_binding'].items():assert sha(p)==h,p
        for p,h in cfg['baseline']['files'].items():assert sha(p)==h,p
        assert sha(out/m/'freeze.json')==first['freeze_hashes'][m]==result['baselines'][m]['freeze_sha256']
        assert sha(out/m/'config.json')==f['config_sha256'] and sha(out/m/'outputs/receipt.json')==f['receipt_sha256']
        assert f['completed_utc']<first['first_reference_access_utc']
        assert len(f['rows'])==len(tables[m])==100 and f['reference_reads']==0
        assert [r['path'] for r in f['opened']]==[p['staged_low'] for p in cohort['rows']]
        assert f['pairs_manifest_sha256']==sha(PAIRS)
    reference_reads=read(out/'reference_reads.json');assert len(reference_reads)==100
    with zipfile.ZipFile(ARCHIVE) as z:
        actual=pairs(z.infolist());assert len(actual)==100
        for i,(p,actual_pair) in enumerate(zip(cohort['rows'],actual)):
            assert all(p[k]==v for k,v in actual_pair.items())
            assert sha(p['staged_low'])==hashlib.sha256(z.read(p['low'])).hexdigest()==p['low_sha256']
            raw=z.read(p['normal']);normal_hash=hashlib.sha256(raw).hexdigest();audit=reference_reads[i]
            assert normal_hash==ours[i]['normal_sha256']==audit['sha256'] and audit['member']==p['normal'] and audit['opened_utc']>=first['first_reference_access_utc']
            with Image.open(io.BytesIO(raw)) as im:y=np.asarray(im.convert('RGB')).astype(np.float64)/255
            y=y.astype(np.float32).astype(np.float64)
            assert ours[i]['index']==i and ours[i]['low']==p['low'] and ours[i]['normal']==p['normal']
            for m in METHODS:
                f=freezes[m]['rows'][i];row=tables[m][i]
                assert f['index']==row['index']==i and f['low']==row['low']==p['low'] and f['normal']==row['normal']==p['normal']
                assert f['low_sha256']==p['low_sha256'] and sha(out/m/f['output'])==f['file_sha256']
                x=np.load(out/m/f['output']);assert x.dtype==np.float32 and x.shape==y.shape==(400,600,3)
                assert np.isfinite(x).all() and 0<=x.min()<=x.max()<=1 and hashlib.sha256(x.tobytes()).hexdigest()==f['output_sha256']
                x=x.astype(np.float64);mse=float(torch.mean((torch.from_numpy(x)-torch.from_numpy(y))**2))
                ps=-10*math.log10(mse);ss=independent_ssim(x,y);delta=ours[i]['psnr']-ps
                values[m].append((ps,ss,delta))
                errors.extend([abs(ps-row['psnr']),abs(ss-row['rgb_ssim']),abs(delta-row['ours_minus_baseline_psnr'])])
    for m,rows in values.items():
        ps,ss,delta=zip(*rows)
        independent=dict(mean_psnr=math.fsum(ps)/100,median_psnr=statistics.median(ps),mean_rgb_ssim=math.fsum(ss)/100,ours_minus_baseline_mean_psnr=math.fsum(delta)/100,ours_minus_baseline_median_psnr=statistics.median(delta),inference_seconds=math.fsum(r['runtime_s'] for r in freezes[m]['rows']))
        errors.extend(abs(v-result['baselines'][m][k]) for k,v in independent.items())
    assert max(errors)<1e-11,max(errors)
    write(out/'verification.json',dict(status='PASS',classification=result['classification'],images_per_method=100,methods=2,independent_reference_reads=100,metric_max_abs_error=max(errors),freeze_hashes=first['freeze_hashes'],cohort_identity=True,output_hashes=True,reference_ordering=True,optimizer_runs=0,model_fits=0,verified_utc=utc()))
    print('Independent verifier PASS',max(errors),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out.resolve())
