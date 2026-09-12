"""Image-header-only T012 manifests; fresh preparation requires passed frozen receipt."""
import argparse
import json
from pathlib import Path
from ttie.residual_data import evaluation_manifest
from ttie.stop_receipt import sha


def prior_ids(logs):
    prior=set()
    for task in ('T004','T005','T006','T007','T008','T009','T011'):
        prior.update(r['image_id'] for r in json.loads((logs/f'{task}_manifest.json').read_text())['images'])
    return prior


def development(pool,logs):
    manifest=evaluation_manifest(pool,prior_ids(logs),count=100,split='development_t012')
    for i,r in enumerate(manifest['images']):r['split']='train_t012_stop' if i<80 else 'calibration_t012_stop'
    manifest['selection']='Numeric ascending; exclude308T004-T011; min-side>=320; first80train,next20calibration'
    return manifest


def fresh(pool,logs,source_manifest,receipt_file,git_verification):
    receipt=json.loads(receipt_file.read_text())
    assert receipt['passes'] and git_verification['git_blob_verified']
    assert sha(receipt_file)==git_verification['receipt_sha256']
    assert sha(source_manifest)==receipt['source_manifest_sha256']
    prior=prior_ids(logs)|{r['image_id'] for r in json.loads(source_manifest.read_text())['images']}
    manifest=evaluation_manifest(pool,prior,split='evaluation_t012')
    manifest['selection']='Numeric ascending; exclude all408prior/source IDs; min-side>=320; first40 after freeze'
    manifest['stopping_receipt_git_verification']=git_verification
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('A','B'),required=True)
    parser.add_argument('--git-verification',type=Path);args=parser.parse_args()
    root=Path('/home/wenchang/asdasdsad/wjq/TTIE');logs=root/'current/research_log';out=root/'shared/t012';out.mkdir(exist_ok=True)
    if args.stage=='A':manifest=development(root/'shared/t008/val2017',logs);name='source_manifest.json'
    else:
        verification=json.loads(args.git_verification.read_text())
        manifest=fresh(root/'shared/t008/val2017',logs,logs/'T012_source_manifest.json',logs/'T012_stopping_receipt.json',verification)
        name='evaluation_manifest.json'
    path=out/name;path.write_text(json.dumps(manifest,indent=2))
    print('IDs:',[r['image_id'] for r in manifest['images']]);print('SHA256:',sha(path))
