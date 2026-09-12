"""T012's explicitly required source/fresh freeze barrier and artifact identities."""
import hashlib
import json
from pathlib import Path
import subprocess
from .stop_quality import load_head,RECIPE
from .stop_trajectory import SCHEMA


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hashes():
    files=sorted(Path('ttie').glob('*.py'))+[Path('scripts/prepare_t012.py'),Path('scripts/run_t012_a6000.sh')]
    return {p.as_posix():sha(p) for p in files}


def create_receipt(head_file,manifest,source_sha,report,fixed,identity,gate_receipt):
    data=json.loads(Path(manifest).read_text());head=load_head(head_file)
    subsets={s:[r for r in data['images'] if r['split']==s] for s in ('train_t012_stop','calibration_t012_stop')}
    return dict(task='T012',passes=report['passes'],source_manifest=data,source_manifest_sha256=sha(manifest),
        split_manifests=subsets,split_manifest_sha256={k:hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest() for k,v in subsets.items()},
        feature_schema=SCHEMA,head_sha256=sha(head_file),normalization=head.normalization(),recipe=RECIPE,
        source_sha=source_sha,source_code_sha256=code_hashes(),stage_a=report,fixed_step_source=fixed,
        model_identity=identity,frozen_gate_receipt=gate_receipt,
        t011_frozen_source='c7ac47a7b43e5cf12f435a8e3be1035898569534')


def verify_receipt(path,head_file,manifest,identity,gate_receipt):
    receipt=json.loads(Path(path).read_text())
    assert receipt['passes'] and receipt['feature_schema']==SCHEMA and receipt['recipe']==RECIPE
    assert receipt['head_sha256']==sha(head_file) and receipt['source_manifest_sha256']==sha(manifest)
    assert receipt['source_code_sha256']==code_hashes()
    assert receipt['model_identity']==identity and receipt['frozen_gate_receipt']==gate_receipt
    assert receipt['normalization']==load_head(head_file).normalization()
    return receipt


def verify_git_receipt(project,commit,receipt_file):
    blob=subprocess.check_output(['git','show',commit+':research_log/T012_stopping_receipt.json'],cwd=project)
    digest=hashlib.sha256(blob).hexdigest()
    assert digest==sha(receipt_file)
    assert json.loads(blob)['passes']
    return dict(commit=commit,receipt_sha256=digest,git_blob_verified=True)
