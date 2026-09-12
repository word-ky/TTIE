"""Run only after a passing Stage A receipt has been committed."""
import argparse
import json
from pathlib import Path
from ttie.residual_data import evaluation_manifest

parser=argparse.ArgumentParser();parser.add_argument('--calibration',required=True,type=Path)
parser.add_argument('--calibration-commit',required=True);args=parser.parse_args()
assert json.loads(args.calibration.read_text())['passes']
root=Path('/home/wenchang/asdasdsad/wjq/TTIE');prior=set()
for task in ('T004','T005','T006','T007','T008','T009'):
    prior.update(r['image_id'] for r in json.loads((root/f'current/research_log/{task}_manifest.json').read_text())['images'])
manifest=evaluation_manifest(root/'shared/t008/val2017',prior)
manifest['calibration_commit']=args.calibration_commit
output=root/'shared/t010';output.mkdir(exist_ok=True)
(output/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('Fresh evaluation IDs:',[r['image_id'] for r in manifest['images']],flush=True)
