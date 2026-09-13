"""T019-C provenance/recipe verification and independent five-file neural replay.

No training implementation import or target/reference artifact open. Reuses the
unchanged, independent T018-D PyTorch reconstruction in a target-free directory.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

root=Path('research_log/T019C_run')
read=lambda name:json.loads((root/name).read_text(encoding='utf-8'))
digest=lambda raw:hashlib.sha256(raw).hexdigest()
receipt=read('selector_frozen.json');pinned=(root/'selector_frozen.sha256').read_text().strip()
assert digest((root/'selector_frozen.json').read_bytes())==pinned
for name,value in receipt['files_sha256'].items():assert digest((root/name).read_bytes())==value
assert sorted(p.name for p in root.glob('*.pt'))==['head_x.pt','head_y.pt']
assert receipt['head_count']==2 and receipt['fit_calls']==dict(x=1,y=1)
assert receipt['training_rows']==receipt['replay_rows']==120 and receipt['epochs']==dict(x=100,y=100)
expected_recipe=dict(input_dim=84,hidden=[64,64],output_dim=3,activation='SiLU',loss='unweighted_cross_entropy',
    optimizer='AdamW',lr=1e-3,weight_decay=1e-4,betas=[.9,.999],optimizer_eps=1e-8,batch_size=256,epochs=100,seed=7,
    checkpoint='final_epoch',device='cpu',normalization='training rows only; population std clamped at 1e-12',normalization_epsilon=1e-12,
    class_order=[.5,.4,.6],training_order='seed7 randperm per epoch; original training row order')
assert receipt['recipe']==expected_recipe
assert receipt['engineering_freeze_only'] and receipt['fresh_qualified'] is False
assert receipt['reference_metrics_opened'] is False and receipt['T018E_data_used'] is False
assert receipt['input_artifact_hashes']['targets']==dict(commit='91f750e871d2c133624bbe4972f3fb3086f25a6c',
    path='research_log/T019A_run/decisions.json',sha256='d874bed74b0ebf68b680c8b6e60c8c5a44c9d82ce3cb7fd5730e16e53a980d45')
cfg=read('config.json')
for key in ('source_sha','source_code_sha256','input_artifact_hashes','recipe','runtime','feature_schema','donor_training_source_sha'):
    assert cfg[key]==receipt[key]
assert cfg['training_rows']==list(range(120))
assert read('normalization.json')==receipt['normalization']
for axis in ('x','y'):
    assert [r['epoch'] for r in read('history_'+axis+'.json')]==list(range(1,101))
inputs={};opened_origins=[]
# Fixed allowlist, deliberately excluding targets. No evaluation inputs exist.
for key in ('score_selection_receipt','score_config','score_selection','accepted_b_config'):
    item=receipt['input_artifact_hashes'][key]
    raw=subprocess.check_output(['git','show',item['commit']+':'+item['path']]);assert digest(raw)==item['sha256']
    inputs[key]=json.loads(raw);opened_origins.append(item['commit']+':'+item['path'])
donor=inputs['accepted_b_config']
assert donor['source_sha']==receipt['donor_training_source_sha'] and donor['recipe']==expected_recipe
for key in ('score_selection_receipt','score_config','score_selection','targets'):
    assert donor['input_artifact_hashes'][key]==receipt['input_artifact_hashes'][key]
for path,value in receipt['source_code_sha256'].items():
    assert digest(Path(path).read_bytes())==value==digest(subprocess.check_output(['git','show',receipt['source_sha']+':'+path]))
for path,value in donor['source_code_sha256'].items():assert receipt['source_code_sha256'][path]==value
assert receipt['inference_code_sha256']=={p:receipt['source_code_sha256'][p] for p in ('ttie/__init__.py','ttie/direction_probe.py','ttie/direction_selector.py')}
assert receipt['feature_schema']==inputs['score_config']['schema']==donor['feature_schema']
assert inputs['score_selection_receipt']['reference_access'] is False
assert inputs['score_selection_receipt']['selection_sha256']==receipt['input_artifact_hashes']['score_selection']['sha256']
assert inputs['score_selection_receipt']['config_sha256']==receipt['input_artifact_hashes']['score_config']['sha256']
scores=inputs['score_selection']['episodes'];assert len(scores)==120
names=['center','x_lower','x_upper','y_lower','y_upper'];indices=[4,1,7,3,5]
assert receipt['feature_order']==cfg['feature_order']==names and cfg['cross_candidate_indices']==indices
assert read('replay_features.json')=={n:[r['features'][i] for r in scores] for n,i in zip(names,indices)}
replay=read('replay_receipt.json')
assert replay['passed'] and replay['target_or_reference_reads'] is False
assert replay['selector_receipt_sha256']==pinned
assert replay['replay_actual_sha256']==digest((root/'replay_actual.json').read_bytes())==receipt['files_sha256']['replay_expected.json']
assert receipt['finalized_utc']<replay['completed_utc']
commands=json.loads(Path('research_log/T019C_commands.json').read_text(encoding='utf-8'))
assert len(commands)==2 and [c['stage'] for c in commands]==['train','replay']
assert all(c['exit_code']==0 for c in commands)
assert commands[0]['started_utc']<=receipt['started_utc']<=receipt['finalized_utc']<=commands[0]['ended_utc']<commands[1]['started_utc']
script=Path('research_log/T018D_replay_verify.py').resolve()
with tempfile.TemporaryDirectory(dir=Path.cwd().parent) as directory:
    for name in ('selector_frozen.json','head_x.pt','head_y.pt','replay_features.json','replay_expected.json'):
        shutil.copy2(root/name,Path(directory)/name)
    proc=subprocess.run([sys.executable,str(script),'--bundle',directory,'--receipt-sha',pinned],cwd=directory,check=True,capture_output=True)
    independent=json.loads(proc.stdout)
assert independent['passed'] and independent['target_or_reference_artifact_reads'] is False
result=dict(passed=True,completed_utc=datetime.now(timezone.utc).isoformat(),selector_receipt_sha256=pinned,
    exact_heads=2,fit_calls=dict(x=1,y=1),literal_recipe=True,all_120_normalization_exact=True,
    target_or_reference_artifact_opened=False,opened_input_origins=opened_origins,
    independent_target_free_bundle=independent,engineering_freeze_only=True,fresh_qualification=False)
Path('research_log/T019C_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))
