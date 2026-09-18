import argparse,math
from ttie.ssim_transfer import rgb_ssim,METRIC
from research_log.T059X.core import *
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out;torch.set_num_threads(1)
f=json.loads((out/'online_freeze.json').read_bytes());assert f['rows']==100
validate_inputs({str(out/n):h for n,h in f['files'].items()});validate_inputs(f['input_hashes_before']);validate_inputs(f['source_bindings'])
records=json.loads((out/'online_records.json').read_bytes());assert len(records)==100
first=utc();assert first>f['utc'];atomic_json(out/'reference_open.json',dict(first_reference_read_utc=first,online_freeze_utc=f['utc']))
validate_inputs(json.loads(Path('research_log/T059X/evaluation_binding.json').read_bytes()))
pairing=json.loads(Path('research_log/T059X/evaluation_pairing.json').read_bytes());assert pairing['split_sha256']=='b88c8347005984b5523b117b52c0c068672fe172eb7c9aa5b60102d350e2d85b';refs=[];rows=[]
for r,s in zip(records,pairing['selected']):
    assert r['low']==s['low'];path=Path(pairing['normal_root'])/s['normal'];assert s['normal'].startswith('Train/Normal/') and sha(path)==s['normal_sha256']
    normal=native_rgb(path).squeeze(0).permute(1,2,0).numpy().astype(float);refs.append(dict(path=str(path),sha256=s['normal_sha256'],utc=utc()))
    t=torch.load(out/r['file'],weights_only=True,map_location='cpu');raw=native_rgb(r['image_file']);assert torch.equal(raw,t['y0'])
    row={k:r[k] for k in ['index','low','norm','active_regions','acted','sha256']}
    for name,key in [('raw','y0'),('ours','y1')]:
        im=t[key].squeeze(0).permute(1,2,0).numpy().astype(float);assert im.shape==normal.shape and np.isfinite(im).all();mse=float(np.mean((im-normal)**2));row[name+'_mse']=mse;row[name+'_psnr']=-10*math.log10(max(mse,1e-12));row[name+'_ssim']=rgb_ssim(im,normal)
    for metric in ['mse','psnr','ssim']:row[metric+'_change']=row['ours_'+metric]-row['raw_'+metric]
    rows.append(row)
atomic_json(out/'reference_reads.json',refs);atomic_json(out/'evaluation_table.json',rows)
validate_inputs({str(out/n):h for n,h in f['files'].items()});validate_inputs(f['input_hashes_before']);validate_inputs({r['path']:r['sha256'] for r in refs})
result=dict(**summarize(rows),online_freeze_utc=f['utc'],first_reference_read_utc=first,metric=METRIC,psnr='RGB float64 MSE over native float32 pixels; PSNR floor1e-12',validation_only=True,official_test_access=0,inference_reference_leakage=0,normal_reads_before_freeze=0,completed_utc=utc());atomic_json(out/'result.json',result);print(json.dumps(result),flush=True)
