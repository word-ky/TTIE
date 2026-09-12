"""T014 metadata-only source/fresh split preparation."""
import argparse
import json
from pathlib import Path
from scripts.prepare_t013 import prior_ids as old_prior
from ttie.residual_data import evaluation_manifest
from ttie.stop_receipt import sha


def prior_ids(logs):
    return old_prior(logs)|{r['image_id'] for r in json.loads((logs/'T013_source_manifest.json').read_text())['images']}


def development(pool,logs):
    manifest=evaluation_manifest(pool,prior_ids(logs),count=100,split='development_t014')
    for i,r in enumerate(manifest['images']):r['split']='train_t014_sobolev' if i<80 else 'calibration_t014_sobolev'
    manifest['selection']='Numeric ascending; exclude508prior IDs; min-side>=320; first80train,next20calibration'
    return manifest


def fresh(pool,logs,source_manifest,receipt_file,git_verification):
    receipt=json.loads(receipt_file.read_text())
    assert receipt['passes'] and git_verification['git_blob_verified'] and git_verification['both_checkpoint_blobs_verified']
    assert sha(receipt_file)==git_verification['receipt_sha256'] and sha(source_manifest)==receipt['source_manifest_sha256']
    prior=prior_ids(logs)|{r['image_id'] for r in json.loads(source_manifest.read_text())['images']}
    manifest=evaluation_manifest(pool,prior,split='evaluation_t014')
    manifest['selection']='Numeric ascending; exclude all608prior/source IDs; min-side>=320; first40 after freeze'
    manifest['energy_receipt_git_verification']=git_verification;return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('A','B'),required=True)
    parser.add_argument('--git-verification',type=Path);args=parser.parse_args()
    root=Path('/home/wenchang/asdasdsad/wjq/TTIE');logs=root/'current/research_log';out=root/'shared/t014';out.mkdir(exist_ok=True)
    if args.stage=='A':manifest=development(root/'shared/t008/val2017',logs);name='source_manifest.json'
    else:
        verification=json.loads(args.git_verification.read_text())
        manifest=fresh(root/'shared/t008/val2017',logs,logs/'T014_source_manifest.json',logs/'T014_energy_receipt.json',verification);name='evaluation_manifest.json'
    path=out/name;path.write_text(json.dumps(manifest,indent=2));print('IDs:',[r['image_id'] for r in manifest['images']]);print('SHA256:',sha(path))
