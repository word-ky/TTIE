import pathlib,json,hashlib,datetime,os,argparse,torch
from research_log.T059Q.storage import tensor_only
from research_log.T059Q.core import decisions,summarize,classify
ROOT=pathlib.Path('/home/wenchang/asdasdsad/wjq/TTIE');E=ROOT/'runs/20260918-023558-ttie-t059e-relative/artifacts/T059E';M=ROOT/'runs/20260918-123128-ttie-t059m-scalar/artifacts/T059M';F=ROOT/'runs/20260918-042133-ttie-t059f2-limit/artifacts/T059F2/result.json';HERE=pathlib.Path('research_log/T059Q')
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def thash(t):return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,v):
 with p.open('w') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
def setup():
 torch.set_num_threads(1);inputs=json.loads((HERE/'inputs.json').read_text());own=json.loads((HERE/'source_binding.json').read_text());assert all(sha(p)==h for p,h in {**inputs,**own}.items());return inputs,own

def decide(out):
 inputs,own=setup();out.mkdir(parents=True,exist_ok=False);split=json.loads((E/'split.json').read_text());ids=split['row_indices']['heldout'];assert len(ids)==1529 and len(set(ids))==1529 and not set(ids)&set(split['row_indices']['outer']);rec=[split['canonical'][i] for i in ids];row=dict(global_=ids,bank=[r['bank_index'] for r in rec],state=[r['state_index'] for r in rec],image=[r['image_id'] for r in rec]);row['global']=row.pop('global_');assert len(set(row['bank']))==80 and len(set(row['image']))==16
 access=[];er=json.loads((E/'heldout_receipt.json').read_text());ma=json.loads((M/'heldout_access.json').read_text());mh=json.loads((M/'initial_final_hashes.json').read_text());assert ma['global_indices']==ids and ma['image_ids']==split['image_ids']['heldout'] and ma['bank_indices']==split['bank_indices']['heldout'];assert er['normalization']==mh['normalization'];assert er['checkpoint_sha256']==inputs[str(E/'head.pt')]
 ep=E/'heldout_row_values.pt';mp=M/'heldout_rows.pt'
 for name,key in [('global_indices','global'),('bank_indices','bank'),('state_indices','state')]:
  t,a=tensor_only(ep,name);access.append(a);assert t.tolist()==row[key] and thash(t)==er['row_tensor_hashes'][name]
 ea,a=tensor_only(ep,'anchor_indices');access.append(a);mm,a=tensor_only(mp,'anchors');access.append(a);assert torch.equal(ea,mm) and thash(ea)==er['row_tensor_hashes']['anchor_indices'] and thash(mm)==ma['tensor_hashes']['anchors']
 for label,path in [('E',ep),('M',mp)]:
  t,a=tensor_only(path,'p');access.append(a);assert len(t)==1529 and torch.isfinite(t).all();row[label]=t.tolist()
  if label=='E':assert thash(t)==er['row_tensor_hashes']['p']
 table,coverage=decisions(row)
 for b in table:
  positions=[i for i,x in enumerate(row['bank']) if x==b['bank']];assert all(int(ea[i])==b['anchor'] for i in positions)
 write(out/'prediction_rows.json',row);write(out/'decisions.json',dict(banks=table,coverage=coverage));write(out/'decisions_persisted.json',dict(persisted_utc=utc(),decisions_sha256=sha(out/'decisions.json'),predictions_sha256=sha(out/'prediction_rows.json'),inputs=inputs,source_binding=own,prediction_storage_reads=access,source_target_tensor_reads=0,normalization=er['normalization'],alignment=dict(images=split['image_ids']['heldout'],rows=1529,banks=80,one_to_one_global_rows=True,one_state0_per_bank=True,M_row_order_basis='accepted heldout_access.global_indices and anchors; source load_side canonical sequence'),sources=dict(E='830e80ba0e1a4d09bce9c9ffd57be162a781d013',M='09a599b5f5da7dc61a5e20716a4ffcf470b0bcda'),prediction_basis='p: unchanged persisted standardized predicted energy; no centering/normalization recomputation',device='cpu'));print('DECISIONS_FROZEN',utc(),coverage,flush=True)

def evaluate(out):
 inputs,own=setup();mark=json.loads((out/'decisions_persisted.json').read_text());assert sha(out/'decisions.json')==mark['decisions_sha256'] and sha(out/'prediction_rows.json')==mark['predictions_sha256'];d=json.loads((out/'decisions.json').read_text());rows=json.loads((out/'prediction_rows.json').read_text());opened=utc();assert mark['persisted_utc']<opened;t,access=tensor_only(E/'heldout_row_values.pt','delta_t');er=json.loads((E/'heldout_receipt.json').read_text());assert thash(t)==er['row_tensor_hashes']['delta_t'];assert len(t)==1529;write(out/'target_rows.json',dict(global_indices=rows['global'],delta_t=t.tolist(),opened_utc=opened,storage=access));table=[];groups={h:[] for h in ['E','M','Q']}
 for b in d['banks']:
  ids=torch.tensor([i for i,k in enumerate(rows['bank']) if k==b['bank']]);minimum=t[ids].min();assert float(t[b['anchor']])==0;entry=dict(**b,oracle_delta_t=float(minimum),anchor_delta_t=float(t[b['anchor']]),policies={})
  for h in groups:
   i=b['a_'+h];v=dict(selected_delta_t=float(t[i]),regret=float(t[i]-minimum),harm_vs_anchor=float(t[i]-t[b['anchor']]));entry['policies'][h]=v;groups[h].append(v)
  table.append(entry)
 stats={h:summarize(v) for h,v in groups.items()};label,criteria=classify(d['coverage']['fraction'],stats['E'],stats['Q']);assert sha(F)=='e34ae1721da1b5ab116b333c8b93731baa6a7eb1fe74ee165daefac055b1623d';f=json.loads(F.read_text());worst={b['bank_index'] for b in f['worst10']};boundary=set(f['partition']['positive_scale_boundary']);regret_tail={b['bank_index'] for b in sorted(f['banks'],key=lambda b:(-b['argmin_regret'],b['bank_index']))[:10]};prior={b['bank_index']:b for b in f['banks']};tails=[]
 for b in table:
  assert b['policies']['E']['regret']==prior[b['bank']]['argmin_regret']
  if b['bank'] in worst|boundary|regret_tail:tails.append(dict(bank=b['bank'],F2_worst10_limit_huber=b['bank'] in worst,F2_top10_original_regret=b['bank'] in regret_tail,F2_boundary=b['bank'] in boundary,agreement=b['agreement'],abstained=b['abstained'],E_state=b['E_state'],M_state=b['M_state'],Q_state=b['Q_state'],policies=b['policies']))
 write(out/'evaluation_table.json',table);before={**inputs,str(F):sha(F)};after={p:sha(p) for p in before};assert before==after and all(sha(p)==h for p,h in own.items());assert sha(out/'decisions.json')==mark['decisions_sha256'];result=dict(task='T059-Q',status='DONE',classification=label,coverage=d['coverage'],statistics=stats,comparative_conditions=criteria,prior_F2_tail= tails,marker=mark,target_opened_utc=opened,target_rows_sha256=sha(out/'target_rows.json'),evaluation_table_sha256=sha(out/'evaluation_table.json'),inputs_before=before,inputs_after=after,authorization='7a1f070ef1a25c13b40820a47f9797b203585c9d',source=json.loads((HERE/'publication.json').read_text())['source'],metric_precision='target differences FP32 as persisted; summaries float64; linear p90; exact oracle hit means target equals minimum including ties',counters={k:0 for k in ['training_runs','optimizer_steps','model_forwards','new_feature_forwards','source_target_reads_before_decision_freeze','outer_supervision_reads','target_domain_access','lolv2_access','official_test_access','inference_reference_leakage']},completed_utc=utc());write(out/'result.json',result);print(json.dumps(dict(classification=label,coverage=d['coverage'],statistics=stats,conditions=criteria)),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['decide','evaluate']);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args();dict(decide=decide,evaluate=evaluate)[a.stage](a.out)
