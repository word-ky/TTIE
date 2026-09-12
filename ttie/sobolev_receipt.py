"""T014 source/fresh freeze identities, including both causal-control heads."""
import hashlib
import json
from pathlib import Path
import subprocess
from .energy_receipt import create_receipt as energy_receipt,code_hashes as energy_hashes
from .energy_model import load_energy,SCHEMA,RECIPE
from .energy_bank import BANK
from .sobolev_source import JACOBIAN
from .sobolev_train import DERIVATIVE_LOSS
from .stop_receipt import sha


def code_hashes():
    return dict(energy_hashes(),**{f:sha(Path(f)) for f in ('scripts/prepare_t014.py','scripts/run_t014_a6000.sh')})


def create_receipt(primary,control,manifest,source_sha,report,identity,gate_receipt,training_manifest):
    result=energy_receipt(primary,manifest,source_sha,report,identity,gate_receipt)
    data=result['source_manifest'];splits={s:[r for r in data['images'] if r['split']==s] for s in ('train_t014_sobolev','calibration_t014_sobolev')}
    result.update(task='T014',split_manifests=splits,
        split_manifest_sha256={k:hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest() for k,v in splits.items()},
        jacobian_convention=JACOBIAN,derivative_loss=DERIVATIVE_LOSS,source_code_sha256=code_hashes(),
        value_only_sha256=sha(control),value_only_normalization=load_energy(control).normalization(),
        source_records_manifest_sha256=sha(training_manifest),source_records_manifest=json.loads(training_manifest.read_text()))
    return result


def verify_receipt(path,primary,control,manifest,identity,gate_receipt):
    r=json.loads(Path(path).read_text())
    assert r['passes'] and r['task']=='T014' and r['feature_schema']==SCHEMA and r['recipe']==RECIPE and r['state_bank']==BANK
    assert r['energy_sha256']==sha(primary) and r['value_only_sha256']==sha(control) and r['source_manifest_sha256']==sha(manifest)
    assert r['source_code_sha256']==code_hashes() and r['fixed_step_source']==16
    assert r['jacobian_convention']==JACOBIAN and r['derivative_loss']==DERIVATIVE_LOSS
    assert r['model_identity']==identity and r['frozen_gate_receipt']==gate_receipt
    assert r['normalization']==load_energy(primary).normalization()==r['value_only_normalization']==load_energy(control).normalization()
    return r


def verify_git_receipt(project,commit,path):
    blob=subprocess.check_output(['git','show',commit+':research_log/T014_energy_receipt.json'],cwd=project)
    digest=hashlib.sha256(blob).hexdigest();r=json.loads(blob);assert digest==sha(path) and r['passes']
    for filename,key in (('T014_energy.pt','energy_sha256'),('T014_value_only.pt','value_only_sha256')):
        weight_blob=subprocess.check_output(['git','show',commit+':research_log/'+filename],cwd=project)
        assert hashlib.sha256(weight_blob).hexdigest()==r[key]
    return dict(commit=commit,receipt_sha256=digest,git_blob_verified=True,both_checkpoint_blobs_verified=True)
