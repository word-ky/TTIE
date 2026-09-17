"""Independent numeric gate and access-boundary verification; no outer supervision."""
import argparse,json
from pathlib import Path
import torch
from ttie.energy_model import load_energy
from research_log.T059E.run import verify_inputs
from research_log.T058A_tangent.core import sha,thash,utc
from research_log.T059A.support import atomic_json
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();read=lambda n:json.loads((a.out/n).read_bytes());m=read('complete.json')
for n,h in m['files'].items():assert sha(a.out/n)==h,n
s=read('summary.json');sp=read('split.json');marker=read('checkpoint_persisted.json');tr=read('train_receipt.json');ev=read('heldout_receipt.json');h=s['inner_heldout'];d=h['detail'];l=h['legacy']
gates=dict(relative_value=h['bank_relative_huber']<=.07650849781930447,detail_positive=d['positive_fraction']>=.75,detail_cosine=d['median_cosine']>=.50,legacy_positive=l['positive_fraction']>=.95,legacy_cosine=l['median_cosine']>=.90)
classification='bank-relative recipe shows nested source-development transfer' if all(gates.values()) else 'bank-relative recipe not supported on fixed nested split'
assert gates=={k:v['passed'] for k,v in s['gates'].items()} and classification==s['classification']==m['classification']
assert len(read('history.json'))==100 and marker['optimizer_steps']==100*((len(sp['row_indices']['train'])+255)//256)
for side,receipt in [('train',tr),('heldout',ev)]:
 access=read(side+'_access.json');allowed=set(sp['row_indices'][side]);outer=set(sp['row_indices']['outer'])
 assert set(access['selected_row_indices'])==allowed and not allowed&outer and [v['bank_index'] for v in access['bank_reads']]==sp['bank_indices'][side]
 for key in ['J','g_R']:
  actual={r['index'] for c in access['cache_reads'] if c['key']==key for r in c['ranges']};assert actual==allowed and not actual&outer
 verify_inputs(access)
 assert receipt['bank_file_hashes_before']==receipt['bank_file_hashes_after'] and receipt['selected_storage_ranges_before']==receipt['selected_storage_ranges_after'] and receipt['tensor_hashes_before']==receipt['tensor_hashes_after']
 rows=torch.load(a.out/(side+'_row_values.pt'),map_location='cpu',weights_only=True);assert receipt['row_tensor_hashes']=={k:thash(v) for k,v in rows.items()}
 assert torch.equal(rows['p']-rows['p'][rows['anchor_indices']],rows['delta_p']) and torch.equal(rows['t']-rows['t'][rows['anchor_indices']],rows['delta_t'])
 assert float(torch.nn.functional.huber_loss(rows['delta_p'],rows['delta_t']))==s['inner_train' if side=='train' else 'inner_heldout']['bank_relative_huber']
assert marker['persisted_utc']<=ev['opened_utc']<=min(b['utc'] for b in read('heldout_access.json')['bank_reads'])
head=load_energy(a.out/'head.pt');assert sha(a.out/'head.pt')==marker['head_sha256'] and {k:thash(v) for k,v in head.state_dict().items()}==tr['final_head_hashes']==ev['final_head_hashes']
atomic_json(a.out/'verification.json',dict(classification=classification,gates_independently_recomputed=gates,all_evidence_hashes_reopened=True,selected_input_ranges_rehashed=True,full_mixed_storage_hashing_performed=False,outer_supervision_reads=0,inner_heldout_supervision_reads_before_checkpoint=0,checkpoint_persisted_utc=marker['persisted_utc'],inner_heldout_opened_utc=ev['opened_utc'],head_sha256=marker['head_sha256'],training_runs=1,optimizer_steps=marker['optimizer_steps'],train_rows=len(sp['row_indices']['train']),heldout_rows=len(sp['row_indices']['heldout']),outer_rows=len(sp['row_indices']['outer']),completed_utc=utc()));print(json.dumps(dict(classification=classification,verification='PASS')),flush=True)
