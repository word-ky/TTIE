import argparse
from scripts.evaluate_t026a import pixels
from ttie.ssim_transfer import rgb_ssim
from research_log.T060CR1.core import *

def main(out):
    setup();f=json.loads((out/'freeze.json').read_bytes());assert len(f['rows'])==100;validate(f['inputs']);validate(f['source_bindings'])
    for r in f['rows']:validate({str(out/f"{r['index']:03d}"/n):h for n,h in r['files'].items()})
    first=utc();assert f['completed_utc']<first;atomic_json(out/'reference_open.json',dict(first_reference_or_baseline_read_utc=first,global_freeze_utc=f['completed_utc']))
    assert sha(BASE/'freeze.json')=='46e667ded785e4f3f8ba341d40d00a7e02ca652e2778fcc7ead1750665161be4'
    old=json.loads((BASE/'freeze.json').read_bytes());manifest=json.loads((HERE/'original_manifest.json').read_bytes());assert sha(HERE/'original_manifest.json')==old['manifest_sha256'];rows=[];opens=[]
    for r,s,b in zip(f['rows'],manifest['selected'],old['rows']):
        assert r['index']==b['index'] and r['low']==s['low']==b['low'];path=NORMAL/s['normal'];assert sha(path)==s['normal_sha256'];normal=pixels(path).astype(np.float32).astype(np.float64);opens.append(dict(path=str(path),sha256=s['normal_sha256'],utc=utc()));row=dict(index=r['index'],low=r['low'])
        for name,p in [('new',out/f"{r['index']:03d}"/'output.pt'),('026',BASE/f"{r['index']:03d}"/'baseline/output.pt'),('036',BASE/f"{r['index']:03d}"/'common/output.pt')]:
            if name!='new':assert sha(p)==b['methods']['baseline' if name=='026' else 'common']['files']['output.pt']['sha256']
            image=torch.load(p,weights_only=True,map_location='cpu')['image'][0].permute(1,2,0).numpy().astype(np.float64);mse=float(np.mean((image-normal)**2));row[name+'_psnr']=float(-10*np.log10(mse));row[name+'_ssim']=rgb_ssim(image,normal)
        for name in ['026','036']:
            for metric in ['psnr','ssim']:row['vs'+name+'_'+metric]=row['new_'+metric]-row[name+'_'+metric]
        rows.append(row)
    atomic_json(out/'metrics.json',rows);atomic_json(out/'normal_reads.json',opens)
    result=dict(**summarize(rows),global_freeze_utc=f['completed_utc'],first_reference_or_baseline_read_utc=first,completed_utc=utc(),official_test_access=0)
    atomic_json(out/'result.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
