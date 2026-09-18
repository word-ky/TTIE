import pathlib,sys,json,hashlib,numpy as np,torch
from research_log.T059Q.storage import tensor_only
out=pathlib.Path(sys.argv[1]);sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();r=json.loads((out/'result.json').read_text());m=r['marker'];d=json.loads((out/'decisions.json').read_text());rows=json.loads((out/'prediction_rows.json').read_text());trow=json.loads((out/'target_rows.json').read_text());ev=json.loads((out/'evaluation_table.json').read_text());assert sha(out/'decisions.json')==m['decisions_sha256'] and sha(out/'prediction_rows.json')==m['predictions_sha256'];assert sha(out/'target_rows.json')==r['target_rows_sha256'] and sha(out/'evaluation_table.json')==r['evaluation_table_sha256'];assert r['inputs_before']==r['inputs_after'] and all(sha(p)==h for p,h in r['inputs_before'].items());assert all(sha(pathlib.Path(r['decision_source_release'])/p)==h for p,h in m['source_binding'].items());assert m['persisted_utc']<trow['opened_utc']==r['target_opened_utc'];assert trow['global_indices']==rows['global'];assert all(x['key'] not in ['t','delta_t','loss'] for x in m['prediction_storage_reads'])
for h,suffix in [('E','/T059E/heldout_row_values.pt'),('M','/T059M/heldout_rows.pt')]:
 path=next(p for p in m['inputs'] if p.endswith(suffix));v,access=tensor_only(path,'p');assert v.tolist()==rows[h]
epath=next(p for p in m['inputs'] if p.endswith('/T059E/heldout_row_values.pt'));target,access=tensor_only(epath,'delta_t');assert target.tolist()==trow['delta_t'];target=target.numpy();bank=np.array(rows['bank']);state=np.array(rows['state']);glob=np.array(rows['global']);stats={h:[] for h in ['E','M','Q']};agree=0;eligible=0
for b,e in zip(d['banks'],ev):
 ids=np.flatnonzero(bank==b['bank']);a=ids[state[ids]==0];assert len(a)==1;a=int(a[0]);selected={}
 for h in ['E','M']:
  scores=np.array(rows[h]);best=np.min(scores[ids]);tied=ids[scores[ids]==best];selected[h]=int(sorted(tied,key=lambda i:(state[i],glob[i]))[0]);assert selected[h]==b['a_'+h]
 ok=selected['E']==selected['M'];selected['Q']=selected['E'] if ok else a;assert b['agreement']==ok and b['abstained']==(not ok) and selected['Q']==b['a_Q'] and b['anchor']==a
 if len(ids)>1:eligible+=1;agree+=ok
 else:assert selected['Q']==a
 for h,i in selected.items():
  assert b[h+'_global']==int(glob[i]) and b[h+'_state']==int(state[i]);v=dict(selected_delta_t=float(target[i]),regret=float(target[i]-target[ids].min()),harm_vs_anchor=float(target[i]-target[a]));assert v==e['policies'][h];stats[h].append(v)
assert len(d['banks'])==80 and len(glob)==1529 and len(set(rows['image']))==16;c=agree/eligible;assert d['coverage']==dict(non_singleton_banks=eligible,agreed_non_singleton=agree,singleton_banks=80-eligible,fraction=c)
summary={}
for h,vals in stats.items():
 a=np.array([v['regret'] for v in vals],dtype=np.float64);harm=np.array([v['harm_vs_anchor'] for v in vals]);q=dict(banks=80,exact_oracle_hit_count=int(sum(a==0)),exact_oracle_hit_rate=float(np.mean(a==0)),mean_regret=float(sum(a)/80),median_regret=float(np.quantile(a,.5)),p90_regret=float(np.quantile(a,.9)),max_regret=float(max(a)),harm_count=int(sum(harm>0)),harm_fraction=float(np.mean(harm>0)))
 for k,v in q.items():assert np.isclose(v,r['statistics'][h][k],rtol=0,atol=1e-15),(h,k)
 summary[h]=q
E=summary['E'];Q=summary['Q'];criteria=dict(lower_mean=Q['mean_regret']<E['mean_regret'],lower_maximum=Q['max_regret']<E['max_regret'],p90_no_worse=Q['p90_regret']<=E['p90_regret'],harm_count_no_increase=Q['harm_count']<=E['harm_count']);label='dual-head consensus is too abstaining to be a useful safety candidate; stop' if c<.5 else 'exact dual-head agreement is not a sufficient target-free safety gate; stop' if not all(criteria.values()) else 'dual-head disagreement is supported as a target-free safety-gate candidate on the already-opened inner-held source diagnostic';assert label==r['classification'] and criteria==r['comparative_conditions'];assert all(v==0 for v in r['counters'].values())
fpath=next(p for p in r['inputs_before'] if '/T059F2/result.json' in p);f=json.loads(pathlib.Path(fpath).read_text());prior={b['bank_index']:b for b in f['banks']};tail={b['bank_index'] for b in f['worst10']}|set(f['partition']['positive_scale_boundary'])|{b['bank_index'] for b in sorted(f['banks'],key=lambda b:(-b['argmin_regret'],b['bank_index']))[:10]};assert tail=={b['bank'] for b in r['prior_F2_tail']}
for b in ev:assert b['E_state']==prior[b['bank']]['predicted_argmin_state_index']
assert max(abs(b['policies']['E']['regret']-prior[b['bank']]['argmin_regret']) for b in ev)==r['historical_F2_max_regret_roundoff']
assert m['persisted_utc']<r['first_target_access']<=r['target_opened_utc']
for b in r['prior_F2_tail']:
 q=next(e for e in ev if e['bank']==b['bank']);assert all(b[k]==q[k] for k in ['agreement','abstained','E_state','M_state','Q_state','policies'])
v=dict(verification='PASS',all1529_prediction_target_rows_and80_bank_decisions_replayed=True,canonical_ties_agreement_abstention_regret_harm_and_classification_verified=True,prior_F2_tail_verified=True,coverage=c,classification=label,result_sha256=sha(out/'result.json'));(out/'verification.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
