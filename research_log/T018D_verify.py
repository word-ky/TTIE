"""Verify provenance bytes, then launch independent replay without target artifacts."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

root=Path('research_log/T018D_run');read=lambda n:json.loads((root/n).read_text(encoding='utf-8'))
digest=lambda b:hashlib.sha256(b).hexdigest()
receipt=read('selector_frozen.json');pinned=(root/'selector_frozen.sha256').read_text(encoding='utf-8').strip()
assert digest((root/'selector_frozen.json').read_bytes())==pinned
for name,h in receipt['files_sha256'].items():assert digest((root/name).read_bytes())==h
for path,h in receipt['source_code_sha256'].items():
    assert digest(Path(path).read_bytes())==h==digest(subprocess.check_output(['git','show','HEAD:'+path]))
inputs={};archive=Path('research_log/T018D_source_inputs')
manifest=json.loads((archive/'manifest.json').read_text(encoding='utf-8'))
for key,item in receipt['input_artifact_hashes'].items():
    filename=key+'.json';assert manifest[filename]==item
    raw=subprocess.check_output(['git','show','HEAD:'+(archive/filename).as_posix()]);assert digest(raw)==item['sha256']
    # Targets are byte-hashed for provenance only, never decoded or passed to replay.
    if key not in ('targets','target_freeze'):inputs[key]=json.loads(raw)
assert receipt['input_artifact_hashes']['targets']['sha256']=='72cd12af095bcef89e461b3e3ec38ec7edad12f86825bc27e91b6509ed3b77c3'
expected_recipe=dict(input_dim=84,hidden=[64,64],output_dim=3,activation='SiLU',loss='unweighted_cross_entropy',
    optimizer='AdamW',lr=1e-3,weight_decay=1e-4,betas=[.9,.999],optimizer_eps=1e-8,batch_size=256,epochs=100,seed=7,
    checkpoint='final_epoch',device='cpu',normalization='training rows only; population std clamped at 1e-12',normalization_epsilon=1e-12,
    class_order=[.5,.4,.6],training_order='seed7 randperm per epoch; original training row order')
assert receipt['recipe']==expected_recipe==inputs['accepted_c_config']['recipe']
assert receipt['head_count']==2 and receipt['fit_calls']==dict(x=1,y=1) and receipt['epochs']==dict(x=100,y=100)
assert receipt['training_rows']==receipt['replay_rows']==120 and receipt['reference_metrics_opened'] is False and receipt['fresh_qualified'] is False
cfg=read('config.json');assert cfg['source_sha']==receipt['source_sha'] and cfg['source_code_sha256']==receipt['source_code_sha256']
assert cfg['input_artifact_hashes']==receipt['input_artifact_hashes'] and cfg['recipe']==expected_recipe
assert read('normalization.json')==receipt['normalization']
assert receipt['feature_schema']==cfg['feature_schema']==inputs['score_config']['schema']==inputs['accepted_c_config']['feature_schema']
assert receipt['feature_schema']['dimension']==len(receipt['feature_schema']['names'])==28
assert cfg['training_rows']==list(range(120))
scores=inputs['score_selection']['episodes'];assert len(scores)==len({r['episode'] for r in scores})==120
assert cfg['training_episode_keys']==[r['episode'] for r in scores]
hard=[[x,y,0.] for x in (.4,.5,.6) for y in (.4,.5,.6)]
assert receipt['candidates']==inputs['score_selection']['candidates']==inputs['score_config']['candidates']==hard
names=['center','x_lower','x_upper','y_lower','y_upper'];indices=[4,1,7,3,5]
assert receipt['feature_order']==cfg['feature_order']==names and cfg['cross_candidate_indices']==indices
cross={n:[r['features'][i] for r in scores] for n,i in zip(names,indices)}
assert cross==read('replay_features.json')
for axis in ('x','y'):
    history=read('history_'+axis+'.json');assert [r['epoch'] for r in history]==list(range(1,101))
assert receipt['inference_code_sha256']=={p:receipt['source_code_sha256'][p] for p in ('ttie/__init__.py','ttie/direction_probe.py','ttie/direction_selector.py')}
assert inputs['accepted_c_config']['source_code_sha256']['ttie/direction_probe.py']==receipt['source_code_sha256']['ttie/direction_probe.py']
commands=json.loads(Path('research_log/T018D_commands.json').read_text(encoding='utf-8'))
assert len(commands)==2 and [c['command'][3] for c in commands]==['train','replay'] and all(c['exit_code']==0 for c in commands)
assert commands[0]['started_utc']<=receipt['started_utc']<=receipt['finalized_utc']<=commands[0]['ended_utc']<commands[1]['started_utc']
replay=read('replay_receipt.json');assert replay['passed'] and replay['selector_receipt_sha256']==pinned and replay['target_or_reference_reads'] is False
assert replay['replay_actual_sha256']==digest((root/'replay_actual.json').read_bytes())==receipt['files_sha256']['replay_expected.json']
script=Path('research_log/T018D_replay_verify.py').resolve()
with tempfile.TemporaryDirectory() as directory:
    for name in ('selector_frozen.json','head_x.pt','head_y.pt','replay_features.json','replay_expected.json'):shutil.copy2(root/name,Path(directory)/name)
    result=subprocess.run([sys.executable,str(script),'--bundle',directory,'--receipt-sha',pinned],cwd=directory,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    independent=json.loads(result.stdout)
assert independent['passed']
result=dict(passed=True,completed_utc=datetime.now(timezone.utc).isoformat(),selector_receipt_sha256=pinned,
    source_hashes=len(receipt['source_code_sha256']),training_input_hashes=len(receipt['input_artifact_hashes']),final_file_hashes=len(receipt['files_sha256']),
    exact_heads=2,formal_train_commands=1,formal_fit_calls=dict(x=1,y=1),epochs=100,accepted_training_rows=120,
    target_bytes_hashed_only_in_provenance_phase=True,target_values_decoded=False,independent_reference_free_replay=independent,
    exact_recipe_and_schema=True,receipt_unchanged=True,fresh_qualification=False,
    provenance_inputs='committed T018D_source_inputs copies; original commit/path/hash retained in manifest')
Path('research_log/T018D_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
