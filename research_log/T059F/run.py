import argparse,json,hashlib,pathlib,datetime,torch
from research_log.T059F.core import decompose
EXPECTED={
'head.pt':'e15e91c4e9be401bcac6d3de039ec2d1ae7141fb6cec40c388d22549652d43d0',
'split.json':'d44f86b7d1920f8f5efc386888ded4c65f543717165b030978f40ad595956621',
'heldout_row_values.pt':'976d8b68e4988623c954230a87577acf5e4d46b5c4583e35bb530fdf1a37b61d',
'heldout_statistics.json':'3673efcfee8872947c53450afc7af948f50d5ad51708768dc3ef2d8150a0122c',
'complete.json':'4dcaea6e44b53a210c92f3901f8a6811ad820185ce0f595fb8be21e2238b3fd9'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',type=pathlib.Path,required=True);ap.add_argument('--output',type=pathlib.Path,required=True);args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
 before={n:sha(args.input/n) for n in EXPECTED};assert before==EXPECTED
 binding=json.loads(pathlib.Path('research_log/T059F/source_binding.json').read_text());assert all(sha(pathlib.Path(n))==h for n,h in binding.items())
 e=json.loads((args.input/'heldout_statistics.json').read_text());s=json.loads((args.input/'split.json').read_text())
 v=torch.load(args.input/'heldout_row_values.pt',map_location='cpu',weights_only=True)
 assert len(v['p'])==1529 and len(v['bank_indices'].unique())==80
 assert v['global_indices'].tolist()==s['row_indices']['heldout']
 replay={'relative_huber':float(torch.nn.functional.huber_loss(v['delta_p'],v['delta_t'])),'detail_positive':e['detail']['positive_fraction'],'detail_median':e['detail']['median_cosine'],'legacy_positive':e['legacy']['positive_fraction'],'legacy_median':e['legacy']['median_cosine']}
 assert list(replay.values())==[.22107574343681335,.868874192237854,.4915534257888794,.9570105671882629,.9057643413543701]
 r=decompose(v)
 assert r['spearman_distribution']==e['spearman_distribution'] and r['regret_distribution']==e['regret_distribution']
 for b,old in zip(r['banks'],e['per_bank']):
  assert b['bank_index']==old['bank_index'] and b['spearman']==old['spearman'] and b['argmin_regret']==old['argmin_regret'] and b['relative_huber']==old['relative_huber']
 after={n:sha(args.input/n) for n in EXPECTED};assert before==after
 assert all(sha(pathlib.Path(n))==h for n,h in binding.items())
 r.update(input_directory=str(args.input),inputs_before=before,inputs_after=after,source_binding=binding,replay=replay,directional_replay='Exact readback from SHA-bound accepted E statistics; row cosines were not persisted. No fresh gradient/forward recomputation.',numeric_convention='CPU float32 state-0 subtraction/dot/scale/product/Huber, as E; stable average ties and first argmin; float64 rank/distribution summary',counters={k:0 for k in ['training_runs','optimizer_steps','new_source_image_opens','reference_gradient_recomputations','new_feature_forwards','new_jacobians','target_domain_access','lolv2_access','official_test_access','inference_reference_leakage','outer_supervision_reads']},completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 (args.output/'result.json').write_text(json.dumps(r,indent=2,allow_nan=False));print(json.dumps({k:r[k] for k in ['status','classification','corrected_huber','invariance_failures','zero_scale_count','scales']}))
if __name__=='__main__':main()
