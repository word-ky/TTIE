from research_log.T059S.core import *
import numpy as np
from PIL import Image
def native_rgb(path):
    with Image.open(path) as im:a=np.asarray(im.convert('RGB')).copy()
    return torch.from_numpy(a).permute(2,0,1).unsqueeze(0).float()/255.
def validate_inputs(inputs):
    for n,h in inputs.items():assert sha(n)==h, n
def install_firewall(root, allowed, output):
    import sys,os
    root=str(Path(root).resolve());allowed={str(Path(p).resolve()) for p in allowed};output=str(Path(output).resolve())
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=str(Path(os.fsdecode(args[0])).resolve())
        if p.startswith(root+'/') and not (p in allowed or p.startswith(output+'/') or p.startswith(root+'/.venv/')):
            raise PermissionError('T059-X inference read/write outside bound low-only inputs: '+p)
    sys.addaudithook(audit)
def classify(coverage,mean,median,wins,ssim):
    if coverage<50:return 'real-domain matched-detail activation is insufficient'
    if mean<0 and median<0 and wins>=55 and ssim>=0:return 'one-step matched-detail direction transfers aggregate real-domain benefit'
    return 'one-step matched-detail direction does not transfer aggregate real-domain benefit'
def summarize(rows):
    keys=['raw_mse','ours_mse','raw_psnr','ours_psnr','raw_ssim','ours_ssim','mse_change','psnr_change','ssim_change']
    stats={k:dict(mean=float(np.mean([r[k] for r in rows])),median=float(np.median([r[k] for r in rows])),p10=float(np.quantile([r[k] for r in rows],.1)),p90=float(np.quantile([r[k] for r in rows],.9))) for k in keys}
    wins=sum(r['mse_change'] < -1e-12 for r in rows);harm=sum(r['mse_change']>1e-12 for r in rows);coverage=sum(r['acted'] for r in rows)
    return dict(rows=len(rows),active_actions=coverage,coverage=coverage/len(rows),improve=wins,harm=harm,tie=len(rows)-wins-harm,statistics=stats,maximum_mse_harm=max(0.,max(r['mse_change'] for r in rows)),classification=classify(coverage,stats['mse_change']['mean'],stats['mse_change']['median'],wins,stats['ssim_change']['mean']))
