"""T013 required source/fresh freeze identities."""
import hashlib
import json
from pathlib import Path
import subprocess
from .stop_receipt import sha
from .energy_model import load_energy,SCHEMA,RECIPE
from .energy_bank import BANK


def code_hashes():
    files=sorted(Path('ttie').glob('*.py'))+[Path('scripts/prepare_t013.py'),Path('scripts/run_t013_a6000.sh')]
    return {p.as_posix():sha(p) for p in files}


def create_receipt(energy_file,manifest,source_sha,report,identity,gate_receipt):
    data=json.loads(Path(manifest).read_text());head=load_energy(energy_file)
    splits={s:[r for r in data['images'] if r['split']==s] for s in ('train_t013_energy','calibration_t013_energy')}
    return dict(task='T013',passes=report['passes'],source_manifest=data,source_manifest_sha256=sha(manifest),
        split_manifests=splits,split_manifest_sha256={k:hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest() for k,v in splits.items()},
        feature_schema=SCHEMA,state_bank=BANK,energy_sha256=sha(energy_file),normalization=head.normalization(),recipe=RECIPE,
        source_sha=source_sha,source_code_sha256=code_hashes(),stage_a=report,fixed_step_source=16,
        model_identity=identity,frozen_gate_receipt=gate_receipt,t011_frozen_source='c7ac47a7b43e5cf12f435a8e3be1035898569534',
        t012_fixed_baseline_source='61b7eb6875343a887b2e55afc96819bab51d60ab')


def verify_receipt(path,energy_file,manifest,identity,gate_receipt):
    receipt=json.loads(Path(path).read_text())
    assert receipt['passes'] and receipt['feature_schema']==SCHEMA and receipt['recipe']==RECIPE and receipt['state_bank']==BANK
    assert receipt['energy_sha256']==sha(energy_file) and receipt['source_manifest_sha256']==sha(manifest)
    assert receipt['source_code_sha256']==code_hashes() and receipt['fixed_step_source']==16
    assert receipt['model_identity']==identity and receipt['frozen_gate_receipt']==gate_receipt
    assert receipt['normalization']==load_energy(energy_file).normalization()
    return receipt


def verify_git_receipt(project,commit,path):
    blob=subprocess.check_output(['git','show',commit+':research_log/T013_energy_receipt.json'],cwd=project)
    digest=hashlib.sha256(blob).hexdigest();assert digest==sha(path) and json.loads(blob)['passes']
    return dict(commit=commit,receipt_sha256=digest,git_blob_verified=True)
