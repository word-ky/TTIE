"""Metadata-only T011 fresh split; run before committing manifest and outcomes."""
import hashlib
import json
from pathlib import Path
from ttie.residual_data import evaluation_manifest


def prepare(pool,logs):
    prior=set()
    for task in ('T004','T005','T006','T007','T008','T009'):
        manifest=json.loads((logs/f'{task}_manifest.json').read_text())
        prior.update(r['image_id'] for r in manifest['images'])
    # T010 Stage A reused exactly T009; no fresh Stage B manifest was created.
    manifest=evaluation_manifest(pool,prior)
    for entry in manifest['images']:entry['split']='evaluation_t011'
    manifest['selection']='Numeric ascending; exclude all T004-T010 inspected image IDs; original min-side>=320; first40'
    manifest['prior_note']='T010 reused T009 development only; no T010 Stage B images inspected'
    return manifest


if __name__=='__main__':
    root=Path('/home/wenchang/asdasdsad/wjq/TTIE')
    manifest=prepare(root/'shared/t008/val2017',root/'current/research_log')
    output=root/'shared/t011';output.mkdir(exist_ok=True)
    path=output/'manifest.json';path.write_text(json.dumps(manifest,indent=2))
    print('IDs:',[r['image_id'] for r in manifest['images']])
    print('SHA256:',hashlib.sha256(path.read_bytes()).hexdigest())
