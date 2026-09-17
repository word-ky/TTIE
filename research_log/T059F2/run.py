import pathlib,json,hashlib,argparse,datetime,torch
from research_log.T059F2.core import audit
E_HASHES={'head.pt':'e15e91c4e9be401bcac6d3de039ec2d1ae7141fb6cec40c388d22549652d43d0','split.json':'d44f86b7d1920f8f5efc386888ded4c65f543717165b030978f40ad595956621','heldout_row_values.pt':'976d8b68e4988623c954230a87577acf5e4d46b5c4583e35bb530fdf1a37b61d','heldout_statistics.json':'3673efcfee8872947c53450afc7af948f50d5ad51708768dc3ef2d8150a0122c'}
F_HASH='8d80308a8e2b071253bc3865f096eadd0f3653efa848e04a0daa703b75b08f6b'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--e',type=pathlib.Path,required=True);a.add_argument('--f',type=pathlib.Path,required=True);a.add_argument('--out',type=pathlib.Path,required=True);a=a.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
 paths={str(a.e/n):h for n,h in E_HASHES.items()};paths[str(a.f)]=F_HASH;before={n:sha(pathlib.Path(n)) for n in paths};assert before==paths
 binding=json.loads(pathlib.Path('research_log/T059F2/source_binding.json').read_text());assert all(sha(pathlib.Path(n))==h for n,h in binding.items())
 v=torch.load(a.e/'heldout_row_values.pt',map_location='cpu',weights_only=True);e=json.loads((a.e/'heldout_statistics.json').read_text());s=json.loads((a.e/'split.json').read_text());f=json.loads(a.f.read_text())
 assert v['global_indices'].tolist()==s['row_indices']['heldout'];assert len(v['p'])==1529
 r=audit(v);assert r['relative_huber']==e['bank_relative_huber']==.22107574343681335
 assert r['spearman_distribution']==e['spearman_distribution'] and r['regret_distribution']==e['regret_distribution']
 assert r['spearman_distribution']['median']==.9356521739130435 and r['regret_distribution']['median']==0
 assert len(r['banks'])==len(f['banks'])==80
 for b,fb,eb in zip(r['banks'],f['banks'],e['per_bank']):
  for k in ['bank_index','rows','numerator','denominator','spearman','spearman_status','argmin_regret','predicted_argmin_state_index']:assert b[k]==fb[k]
  for k in ['bank_index','spearman','spearman_status','argmin_regret']:assert b[k]==eb[k]
  assert b['limit_huber']==fb['corrected_huber']
 assert r['partition']['positive_scale_boundary']==[230,280,305]
 deg=r['partition']['scale_unidentified_degenerate'];assert len(deg)==17 and deg==[b['bank_index'] for b in f['banks'] if b['denominator']==0];assert all(b['rows']==1 for b in r['banks'] if b['bank_index'] in deg)
 assert abs(r['limit_huber']-.21471332013607025)<=2e-7+2e-6*.21471332013607025
 after={n:sha(pathlib.Path(n)) for n in paths};assert after==before;assert all(sha(pathlib.Path(n))==h for n,h in binding.items())
 r.update(status='DONE',inputs_before=before,inputs_after=after,source_binding=binding,directional_accepted_readback={k:e[k] for k in ['detail','legacy']},directional_note='Hash-bound E aggregate readback only; no cosine vectors persisted, no gradient regeneration',replay='All F numerators/denominators and per-bank loss values exact; E relative value and original rank/regret exact',ordering_proof='For a>0, sign(a*(p_i-p_j))=sign(p_i-p_j); ties and earliest argmin preserved. At a->0+ replay original ordering, never order the zero loss-limit vector. Degenerate banks contribute loss but no scale evidence.',numeric_convention='Frozen CPU float32 residual/dot/Huber; float64 rank/distribution; tolerance atol2e-7 rtol2e-6 inherited F verifier',counters={k:0 for k in ['training_runs','optimizer_steps','model_forwards','new_source_image_opens','reference_gradient_recomputations','new_feature_forwards','new_jacobians','outer_supervision_reads','target_domain_access','lolv2_access','official_test_access','inference_reference_leakage']},completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 (a.out/'result.json').write_text(json.dumps(r,indent=2,allow_nan=False));print(json.dumps({k:r[k] for k in ['status','classification','limit_huber','threshold_margin','partition']}))
if __name__=='__main__':main()
