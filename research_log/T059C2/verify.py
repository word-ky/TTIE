"""Independently recompute C2 gates and reopen persisted evidence; no training."""
import argparse,json
from pathlib import Path
from ttie.energy_model import load_energy
from research_log.T058A_tangent.core import sha,thash,utc
from research_log.T059A.support import atomic_json
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();read=lambda n:json.loads((a.out/n).read_bytes())
m=read('complete.json')
for n,h in m['files'].items():assert sha(a.out/n)==h,n
s=read('summary.json');split=read('split.json');tr=read('train_receipt.json');er=read('evaluation_receipt.json');access=read('training_access.json');ea=read('heldout_access.json');marker=read('checkpoint_persisted.json')
l=s['heldout']['legacy'];d=s['heldout']['detail'];values={'detail_positive':(d['positive_fraction'],.75),'detail_cosine':(d['median_cosine'],.50),'legacy_positive':(l['positive_fraction'],.95),'legacy_cosine':(l['median_cosine'],.90),'value_huber':(l['value_huber'],.07650849781930447)}
gates={k:(v<=t if k=='value_huber' else v>=t) for k,(v,t) in values.items()}
classification='image-held-out dual-tangent generalization supported' if all(gates.values()) else 'image-held-out dual-tangent generalization not supported under fixed split'
assert gates=={k:g['passed'] for k,g in s['gates'].items()} and classification==s['classification']==er['classification']==m['classification']
assert len(read('history.json'))==100 and tr['training_runs']==1 and tr['optimizer_steps']==100*((tr['training_rows']+255)//256)
assert set(access['loaded_reference_indices'])==set(split['row_indices']['train']) and set(ea['loaded_reference_indices'])==set(split['row_indices']['heldout'])
assert not set(access['loaded_reference_indices'])&set(ea['loaded_reference_indices'])
assert {r['index'] for c in access['reference_storage_reads'] for r in c['ranges']}==set(split['row_indices']['train'])
assert [b['bank_index'] for b in access['supervision_banks']]==split['bank_indices']['train']
assert marker['persisted_utc']<=er['heldout_evaluation_opened_utc']<=min([b['utc'] for b in ea['supervision_banks']]+[c['utc'] for c in ea['reference_storage_reads']])
head=load_energy(a.out/'head.pt');assert sha(a.out/'head.pt')==marker['head_sha256']==er['head_sha256'] and {k:thash(v) for k,v in head.state_dict().items()}==tr['final_head_hashes']==er['final_head_hashes']
for rec in [tr,er]:
 assert rec['asset_hashes_before']==rec['asset_hashes_after']
 for n,h in rec['asset_hashes_after'].items():assert sha(Path(n))==h,n
verification=dict(classification=classification,numeric_gates_independently_recomputed=gates,all_file_hashes_reopened=True,head_reopened=True,head_sha256=sha(a.out/'head.pt'),history_epochs=100,optimizer_steps=tr['optimizer_steps'],training_reference_rows=len(access['loaded_reference_indices']),heldout_reference_rows=len(ea['loaded_reference_indices']),training_supervision_bank_indices=split['bank_indices']['train'],heldout_supervision_reads_before_checkpoint=0,checkpoint_persisted_utc=marker['persisted_utc'],heldout_evaluation_opened_utc=er['heldout_evaluation_opened_utc'],completed_utc=utc())
atomic_json(a.out/'verification.json',verification);print(json.dumps(verification),flush=True)
