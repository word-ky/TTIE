"""Separate post-freeze source-only oracle-state/selection-regret diagnostic."""
import argparse
from ttie.natural import load_image
from research_log.T060DR2.preflight import *

def summarize(rows):
    keys=['selected_delta','oracle_delta','regret_delta']
    metrics={k:stats([r[k] for r in rows]) for k in keys}
    counts={k:dict(win=sum(r[k]>0 for r in rows),equal=sum(r[k]==0 for r in rows),loss=sum(r[k]<0 for r in rows)) for k in keys}
    gates=dict(oracle_mean=metrics['oracle_delta']['mean']>=.15,oracle_median=metrics['oracle_delta']['median']>0,oracle_wins=counts['oracle_delta']['win']>=36,regret_mean=metrics['regret_delta']['mean']>=.10)
    label='T059 gain-slice creates better source trajectories but T014 scalar selection leaves material value unrealized' if all(gates.values()) else 'T014-selection mismatch is not a sufficient explanation for the T060-C-R1 near-miss'
    return dict(rows=len(rows),metrics=metrics,counts=counts,gates=gates,classification=label)

def main(out):
    setup();f=json.loads((out/'freeze.json').read_bytes());assert len(f['rows'])==60
    validate(f['inputs']);validate(f['source_bindings'])
    for r in f['rows']:
        for name,m in r['methods'].items():validate({str(out/f"{r['order']:03d}"/name/n):h for n,h in m['files'].items()})
    first=utc();assert f['completed_utc']<first;atomic_json(out/'reference_open.json',dict(first_source_clean_read_utc=first,global_freeze_utc=f['completed_utc']))
    manifest=Path('research_log/T014_source_manifest.json');assert sha(manifest)=='4dbf8c604bc44574a5823ec8e22142c8ac39574b6103c4bbd0a69647f15a2257'
    sources={r['image_id']:r for r in json.loads(manifest.read_bytes())['images'] if r['split']=='train_t014_sobolev'};clean={};opens=[];rows=[]
    for r in f['rows']:
        i=r['image_id']
        if i not in clean:
            s=sources[i];path=ROOT/'shared/t008/val2017'/s['filename'];assert sha(path)==s['sha256'];clean[i]=load_image(path).double();opens.append(dict(image_id=i,path=str(path),sha256=s['sha256'],utc=utc()))
        row={k:r[k] for k in ['order','index','bank_index','image_id']}
        for name in ['A','B']:
            folder=out/f"{r['order']:03d}"/name;images=torch.load(folder/'images.pt',weights_only=True,map_location='cpu');assert len(images)==41 and images[0].shape==clean[i].shape
            mse=[float((im.double()-clean[i]).square().mean()) for im in images];psnr=[float(-10*np.log10(x)) for x in mse];assert all(np.isfinite(psnr))
            selected=r['methods'][name]['selected_step'];best=max(range(41),key=lambda j:(psnr[j],-j))
            row[name]=dict(selected_step=selected,selected_psnr=psnr[selected],oracle_step=best,oracle_psnr=psnr[best],regret=psnr[best]-psnr[selected],mse=mse,psnr=psnr)
        for k,v in [('selected_delta','selected_psnr'),('oracle_delta','oracle_psnr'),('regret_delta','regret')]:row[k]=row['B'][v]-row['A'][v]
        rows.append(row)
    atomic_json(out/'source_clean_reads.json',opens);atomic_json(out/'metrics.json',rows)
    atomic_json(out/'result.json',dict(**summarize(rows),global_freeze_utc=f['completed_utc'],first_source_clean_read_utc=first,source_clean_images=len(clean),completed_utc=utc(),lolv2_access=0,official_test_access=0));print(json.dumps(summarize(rows)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
