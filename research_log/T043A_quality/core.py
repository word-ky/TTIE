"""Fixed paired quality summaries; no state or inference code."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,numpy as np
LABEL='REFERENCE_EVALUATION_ONLY'
COHORT='279c74b335999d6631d436c79ea80e9e2cccfc4b97cb7c0b992829d04cfaf40b'
FREEZES=dict(baseline='b2612273d37eb982c40736d5d3434cfa717666a8cad70e18574f41968b08d64b',candidate='a5b0799ee370bfd482393fce59895d904008cee9c577d85b385c3ce19dd92c36')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def thash(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def classify(psnr,ssim):return 'matched-gain early-state quality bridge supported' if psnr>=.5 and ssim>=0 else 'matched-gain early-state quality bridge not supported / mixed'
def aggregate(rows):
    result=dict(pairs=len(rows),means={k:float(np.mean([r[k] for r in rows])) for k in ['baseline_psnr','candidate_psnr','baseline_ssim','candidate_ssim']},paired={})
    for metric in ['psnr','ssim']:
        key='delta_'+metric;v=np.array([r[key] for r in rows]);ordered=sorted(rows,key=lambda r:(r[key],r['index']))
        result['paired'][metric]=dict(mean=float(v.mean()),median=float(np.median(v)),p10=float(np.quantile(v,.1,method='linear')),p90=float(np.quantile(v,.9,method='linear')),positive=int((v>0).sum()),negative=int((v<0).sum()),zero=int((v==0).sum()),worst=[dict(index=r['index'],low=r['low'],delta=r[key]) for r in ordered[:5]],best=[dict(index=r['index'],low=r['low'],delta=r[key]) for r in ordered[-5:][::-1]])
    result['classification']=classify(result['paired']['psnr']['mean'],result['paired']['ssim']['mean']);return result
