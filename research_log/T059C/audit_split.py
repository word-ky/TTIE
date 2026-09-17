"""T059-C manifest-only split audit. No tensor, target, image or model loading."""
from pathlib import Path
import json,hashlib,datetime,collections
root=Path('/home/wenchang/asdasdsad/wjq/TTIE')
p=root/'runs/20260912-181047-ttie-t014-stage-a-repaired/artifacts/audit/training_manifest.json'
raw=p.read_bytes();digest=hashlib.sha256(raw).hexdigest();assert digest=='92fd60d01ca0780880d724464c4d0a76ba1721ab5b8a2d054d4035a141673125'
manifest=json.loads(raw);rows=[];banks=[];cursor=0
for i,b in enumerate(manifest):
 side='heldout' if i%5==0 else 'train'
 banks.append(dict(bank_index=i,image_id=b['image_id'],directory=b['directory'],states=b['states'],side=side,row_start=cursor,row_stop=cursor+b['states']))
 rows.extend(dict(index=cursor+j,bank_index=i,image_id=b['image_id'],side=side) for j in range(b['states']));cursor+=b['states']
held=[b for b in banks if b['side']=='heldout'];train=[b for b in banks if b['side']=='train'];hi=sorted(set(b['image_id'] for b in held));ti=sorted(set(b['image_id'] for b in train));overlap=sorted(set(hi)&set(ti))
checks=dict(exactly_80_manifest_banks=len(banks)==80,exactly_16_heldout_banks=len(held)==16,exactly_64_train_banks=len(train)==64,zero_image_overlap=not overlap,all_7346_rows_assigned_once=len(rows)==7346 and [r['index'] for r in rows]==list(range(7346)))
result=dict(task='T059-C',status='BLOCKED',classification='split/binding mismatch; training not executed',authorization='8fe42e83182a57e490406331dcacc981343542e1',manifest_path=str(p),manifest_sha256=digest,manifest_banks=len(banks),unique_images=len(set(b['image_id'] for b in banks)),banks_per_image_frequency=dict(collections.Counter(collections.Counter(b['image_id'] for b in banks).values())),heldout_bank_indices=[b['bank_index'] for b in held],train_bank_indices=[b['bank_index'] for b in train],heldout_image_ids=hi,train_image_ids=ti,overlap_image_ids=overlap,heldout_rows=sum(b['states'] for b in held),train_rows=sum(b['states'] for b in train),total_rows=len(rows),checks=checks,banks=banks,canonical_rows=rows,training_runs=0,optimizer_steps=0,supervision_tensor_loads=0,heldout_supervision_reads_before_checkpoint=0,new_source_image_opens=0,reference_gradient_recomputations=0,new_feature_forwards=0,target_domain_access=0,lolv2_access=0,official_test_access=0,inference_reference_leakage=0,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),manifest_unchanged=hashlib.sha256(p.read_bytes()).hexdigest()==digest,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
assert not all(checks.values())
out=root/'shared/t059c';out.mkdir(exist_ok=True);(out/'T059C_split_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ['banks','canonical_rows','train_bank_indices','heldout_bank_indices','heldout_image_ids','train_image_ids','overlap_image_ids']}))
