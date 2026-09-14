"""REFERENCE_DIAGNOSTIC_ONLY summaries of immutable T036 trajectories."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
LABEL='REFERENCE_DIAGNOSTIC_ONLY'
COHORT='279c74b335999d6631d436c79ea80e9e2cccfc4b97cb7c0b992829d04cfaf40b'
PRIOR_FREEZE='46e667ded785e4f3f8ba341d40d00a7e02ca652e2778fcc7ead1750665161be4'
PRIOR_METRICS='cdbd7fec76194153db645133d5d35dd43f6f9d852ebbd6674756d77d1ad7aee0'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def stats(v):
    return dict(count=len(v),mean=float(np.mean(v)),median=float(np.median(v)),p05=float(np.quantile(v,.05)),p95=float(np.quantile(v,.95)),min=float(min(v)),max=float(max(v)))
def image_diagnostic(index,low,quality,selected):
    row=dict(index=index,low=low,baseline_selected_step=selected['baseline'],common_selected_step=selected['common'])
    for metric in ['psnr','ssim']:
        values=np.asarray(quality['common'][metric]);base=float(quality['baseline'][metric][selected['baseline']]);step=selected['common']
        best=int(np.argmax(values));earlier=np.flatnonzero(values[:step]>=base)
        row.update({f'baseline_selected_{metric}':base,f'common_selected_{metric}':float(values[step]),
            f'common_reference_best_{metric}_step':best,f'common_reference_best_{metric}':float(values[best]),
            f'{metric}_headroom':float(values[best]-values[step]),f'prior_{metric}_loss':bool(values[step]<base),
            f'earlier_{metric}_reaches_baseline':bool(len(earlier)),f'earliest_{metric}_rescue_step':int(earlier[0]) if len(earlier) else None})
    return row
def verdict(mean_headroom,rescued_psnr_losses):
    return 'strong late-selection headroom' if mean_headroom>=.75 and rescued_psnr_losses>=15 else 'limited/mixed late-selection headroom'
