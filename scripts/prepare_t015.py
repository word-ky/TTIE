"""Metadata-only fresh T015 split; no training/calibration or outcome access."""
import argparse
import json
from pathlib import Path
from scripts.prepare_t014 import prior_ids as old_prior
from ttie.residual_data import evaluation_manifest
from ttie.stop_receipt import sha


def fresh(pool,logs):
    prior=old_prior(logs)
    for name in ('T014_source_manifest.json','T014_manifest.json'):
        prior|={r['image_id'] for r in json.loads((logs/name).read_text())['images']}
    assert len(prior)==648
    m=evaluation_manifest(pool,prior,split='evaluation_t015')
    m['selection']='Numeric ascending; exclude all648inspected IDs; original min-side>=320; first40; six primary conditions'
    r=json.loads((logs/'T014_energy_receipt.json').read_text())
    m['frozen_t014_receipt_sha256']=sha(logs/'T014_energy_receipt.json')
    m['frozen_sobolev_sha256']=r['energy_sha256'];return m


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--images',type=Path,required=True)
    p.add_argument('--logs',type=Path,default=Path('research_log'));p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    m=fresh(a.images,a.logs);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(m,indent=2)+'\n')
    print(json.dumps(dict(ids=[r['image_id'] for r in m['images']],sha256=sha(a.output),excluded=len(m['excluded_prior_ids'])),indent=2))
