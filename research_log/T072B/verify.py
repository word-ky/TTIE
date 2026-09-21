"""Audit metadata, hashes and one-shot attempt receipts without inference."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,torch
from PIL import Image
from research_log.T063A.common import sha,thash,utc,write
def main(low,out):
    d=Path('research_log/T072B');get=lambda n:json.loads((d/n).read_bytes())
    p=get('pairs_manifest.json');choice=get('smoke_selection.json');acq=get('input_receipt.json')
    input_files=get('input_metadata.json')['files'];gt_files=get('gt_metadata.json')['files']
    a={x['name']:x for x in input_files};b={x['name']:x for x in gt_files}
    assert len(a)==len(b)==len(p['rows'])==150 and set(a)==set(b)
    assert p['rows']==[dict(name=n,input=a[n],gt=b[n]) for n in sorted(a)]
    assert choice['name']==sorted(a)[0]==low.name and choice['file_id']==a[low.name]['id']
    assert choice['declared_utc']<acq['acquired_utc'] and choice['pixels_read']==0
    assert sha(d/'pairs_manifest.json')==choice['pairs_manifest_sha256']
    assert sha(low)==acq['sha256'] and low.stat().st_size==int(a[low.name]['size'])==acq['bytes']
    with Image.open(low) as im:assert (im.width,im.height)==(3840,2160)
    methods=['ours','retinexformer','snr_aware'];records=[]
    for method in methods:
        file=out/method/'receipt.json'
        if not file.exists():break
        r=json.loads(file.read_bytes());assert r['attempts']==1 and r['reference_reads']==r['model_fits']==0
        assert r['opened']==[str(low)] and r['input_sha256']==acq['sha256'] and r['post_binding_unchanged']
        for path,h in r['source_binding'].items():assert sha(path)==h,path
        if r['status']=='PASS':
            assert sha(out/method/r['output'])==r['output_file_sha256']
            if method=='ours':
                x=torch.load(out/method/r['output'],weights_only=True,map_location='cpu');assert tuple(x.shape)==(1,3,2160,3840) and torch.isfinite(x).all() and thash(x)==r['output_hash']
            else:
                x=np.load(out/method/r['output']);assert x.shape==(2160,3840,3) and np.isfinite(x).all() and hashlib.sha256(x.tobytes()).hexdigest()==r['output_hash']
        else:assert r['status']=='BLOCKED' and (out/method/'traceback.txt').exists()
        records.append(dict(method=method,status=r['status'],receipt_sha256=sha(file)))
        if r['status']=='BLOCKED':break
    assert records
    classification='UHDLL_NATIVE_PREFLIGHT_PASS' if len(records)==3 and all(r['status']=='PASS' for r in records) else 'BLOCKED'
    write(out/'verification.json',dict(status='PASS',classification=classification,pairs=150,selection=low.name,metadata_and_hash_audit=True,reference_reads=0,model_fits=0,inference_reruns=0,methods=records,unrun=methods[len(records):],verified_utc=utc()))
    print('Independent saved-artifact audit PASS',classification,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--low',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.low.resolve(),a.out.resolve())
