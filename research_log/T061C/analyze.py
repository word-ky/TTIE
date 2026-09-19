"""Freeze immutable evaluation intent, then evaluate exactly common step 11."""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
from datetime import datetime, timezone

SOURCE_MANIFEST = 'research_log/T061B/result/manifest.json'
SOURCE_SHA = '34a15a13f0c94b07a7eef870efee9f28da51e30abc7f440a342a62da5c0cd1b5'
DEV_COMMIT = 'c0d84b1d3c7e6af186c28ca736d6ac2bc752d35c'
INPUTS = {
    'research_log/T037A_result/per_step.csv': dict(git_blob='0c86ef3aaba2688eb10586018a499b8ef9d5ab7c', sha256='b785767669261999212eb19e563d5f858c7ecc2598ce17a82ec0eaf3a4ddf270'),
    'research_log/T037A_result/per_image.csv': dict(git_blob='39b7e663d18dc8accc6d01c40cbc801f808e96f3', sha256='5676541d245fdb41c54a543cf88a79ca37640eea94432af0665cca2ddce2277c'),
}
GATES = dict(mean_psnr_vs_t036_ge=0.20, median_psnr_vs_t036_gt=0.0,
    regressions_vs_t026_le=29, worst_psnr_vs_t026_ge=-5.614, mean_ssim_vs_t036_ge=-0.001)
POSITIVE = 'source-chosen fixed stopping is a transferable T036 selector improvement'
NEGATIVE = 'a single source-chosen fixed stopping step does not transfer sufficiently'

def utc():
    return datetime.now(timezone.utc).isoformat()

def sha(data):
    return hashlib.sha256(data).hexdigest()

def save(path, value):
    data = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    with path.open('xb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    assert path.read_bytes() == data
    return sha(data)

def bound(root, name):
    b = (root / name).read_bytes()
    assert sha(b) == INPUTS[name]['sha256']
    assert hashlib.sha1(b'blob ' + str(len(b)).encode() + b'\0' + b).hexdigest() == INPUTS[name]['git_blob']
    return b

def summarize(rows):
    p = [r['delta_psnr_t036'] for r in rows]
    s = [r['delta_ssim_t036'] for r in rows]
    b = [r['delta_psnr_t026'] for r in rows]
    counts = lambda v: dict(improve=sum(x>0 for x in v), regress=sum(x<0 for x in v), tie=sum(x==0 for x in v))
    stats = dict(mean_psnr_vs_t036=statistics.fmean(p), median_psnr_vs_t036=statistics.median(p),
        mean_ssim_vs_t036=statistics.fmean(s), mean_psnr_vs_t026=statistics.fmean(b), worst_psnr_vs_t026=min(b),
        counts_vs_t036=counts(p), counts_vs_t026=counts(b))
    gates = dict(mean_psnr=stats['mean_psnr_vs_t036']>=.20, median_psnr=stats['median_psnr_vs_t036']>0,
        regressions=stats['counts_vs_t026']['regress']<=29, worst_psnr=stats['worst_psnr_vs_t026']>=-5.614,
        mean_ssim=stats['mean_ssim_vs_t036']>=-.001)
    return dict(stats=stats, gates=gates, classification=POSITIVE if all(gates.values()) else NEGATIVE,
        verdict='PASS' if all(gates.values()) else 'NEGATIVE')

def paired(steps, images):
    assert len(images)==100 and len({r['low'] for r in images})==100
    assert [int(r['index']) for r in images]==list(range(100))
    common=[r for r in steps if r['method']=='common']
    assert len(common)==4100
    assert [(int(common[i*41]['index']),common[i*41]['low']) for i in range(100)]==[(int(r['index']),r['low']) for r in images]
    lookup={}
    for r in steps:
        key=(int(r['index']),r['low'],r['method'],int(r['step']))
        assert key not in lookup
        assert all(math.isfinite(float(r[k])) for k in ['psnr','ssim'])
        lookup[key]=r
    rows=[]
    for i, image in enumerate(images):
        c=common[i*41:(i+1)*41]
        assert [int(r['step']) for r in c]==list(range(41))
        assert all(r['low']==image['low'] and int(r['index'])==i for r in c)
        out=dict(index=i,low=image['low'],step=11)
        for method in ['common','baseline']:
            selected=int(image[method+'_selected_step'])
            rec=lookup[(i,image['low'],method,selected)]
            for metric in ['psnr','ssim']:
                assert float(rec[metric])==float(image[method+'_selected_'+metric])
        for metric in ['psnr','ssim']:
            out['step11_'+metric]=float(c[11][metric])
            out['identity_'+metric]=float(c[0][metric])
            out['t036_'+metric]=float(image['common_selected_'+metric])
            out['t026_'+metric]=float(image['baseline_selected_'+metric])
            assert all(math.isfinite(out[k+'_'+metric]) for k in ['step11','identity','t036','t026'])
            for base in ['t036','t026']:
                out['delta_'+metric+'_'+base]=out['step11_'+metric]-out[base+'_'+metric]
        rows.append(out)
    return rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['prepare','evaluate'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--script-commit');p.add_argument('--script-sha256')
    a=p.parse_args();root=a.root.resolve();out=a.out.resolve()
    raw=(root/SOURCE_MANIFEST).read_bytes();assert sha(raw)==SOURCE_SHA and json.loads(raw)['k_star']==11
    if a.mode=='prepare':
        out.mkdir(parents=True,exist_ok=False)
        for name in INPUTS:bound(root,name)  # Hash bytes only; no development fields parsed.
        intent=dict(task='T061-C',k_star=11,source_manifest_sha256=SOURCE_SHA,development_commit=DEV_COMMIT,
            inputs=INPUTS,gates=GATES,gate_preregistration='5da5a57e0b481a98d5087aacf7afde8771f80217',
            script_commit=a.script_commit,script_sha256=a.script_sha256,frozen_utc=utc(),
            policy='One fixed common step 11 for all 100 images; offline evaluation only; no second candidate.')
        h=save(out/'intent.json',intent);save(out/'intent_hash.json',dict(sha256=h));print('INTENT_FROZEN',h)
        return
    intent_bytes=(out/'intent.json').read_bytes();intent=json.loads(intent_bytes)
    ih=sha(intent_bytes);assert ih==json.loads((out/'intent_hash.json').read_bytes())['sha256']
    assert intent['inputs']==INPUTS and intent['gates']==GATES and intent['k_star']==11 and intent['source_manifest_sha256']==SOURCE_SHA
    first=utc();assert first>intent['frozen_utc']
    save(out/'quality_read_start.json',dict(first_development_quality_read_utc=first,intent_sha256=ih))
    tables=[list(csv.DictReader(io.StringIO(bound(root,n).decode()))) for n in INPUTS]
    rows=paired(*tables);save(out/'paired.json',rows)
    result=dict(**summarize(rows),images=100,k_star=11,intent_sha256=ih,
        intent_frozen_utc=intent['frozen_utc'],first_development_quality_read_utc=first,completed_utc=utc(),
        new_optimizer_runs=0,new_render_runs=0,official_test_access=0,cross_dataset_access=0)
    save(out/'result.json',result);print(json.dumps(result))

if __name__=='__main__':main()
