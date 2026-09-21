import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np

HERE=Path('research_log/T071B')
PAIRS=Path('research_log/T071A/evidence/pairs_manifest.json')
OURS=Path('research_log/T071A/evidence/per_image_metrics.json')
ARCHIVE=Path('/media/wenchang/F/wjq/TTIE/shared/t022a/LOL-v2.zip')
METHODS=('retinexformer','snr_aware')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_bytes())
def write(path,obj):Path(path).write_bytes((json.dumps(obj,indent=2)+'\n').encode())
def summary(rows):
    return dict(images=len(rows),mean_psnr=float(np.mean([r['psnr'] for r in rows])),median_psnr=float(np.median([r['psnr'] for r in rows])),mean_rgb_ssim=float(np.mean([r['rgb_ssim'] for r in rows])),ours_minus_baseline_mean_psnr=float(np.mean([r['ours_minus_baseline_psnr'] for r in rows])),ours_minus_baseline_median_psnr=float(np.median([r['ours_minus_baseline_psnr'] for r in rows])))
