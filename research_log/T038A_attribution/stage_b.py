"""Isolated reference-MSE gradients on states frozen by low-only Stage A."""
from core import *
import argparse,csv,time
from PIL import Image
from scripts.run_t036a import initialize
from ttie.common_gain import CommonRegion2
from ttie.lolv2_gamma_core import native_rgb
p=argparse.ArgumentParser()
for k in ['stage-a','manifest','accepted','low-root','normal-root','out']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();initialize();start=time.perf_counter();began=utc()
assert sha(a.manifest)==COHORT and sha(a.accepted/'metrics.csv')==PRIOR_METRICS
f=json.loads((a.stage_a/'freeze.json').read_bytes());cohort=json.loads(a.manifest.read_bytes())['selected']
assert f['cohort_sha256']==COHORT and f['normal_decodes']==f['optimizer_updates']==f['selection_changes']==0
for name,h in f['source_binding'].items():assert sha(name)==h
for row in f['rows']:assert sha(a.stage_a/row['file'])==row['sha256']
prior=list(csv.DictReader((a.accepted/'metrics.csv').open()));loss={r['low'] for r in prior if float(r['delta_psnr'])<0};assert len(loss)==29
opened=[];normal_allowed={str((a.normal_root/r['normal']).resolve()) for r in cohort};low_allowed={str((a.low_root/r['low']).resolve()) for r in cohort};original=Image.open
def paired(path,*args,**kwargs):
    name=str(Path(path).resolve());assert name in normal_allowed|low_allowed
    opened.append(dict(path=name,utc=utc(),kind='normal' if name in normal_allowed else 'low'));return original(path,*args,**kwargs)
Image.open=paired;a.out.mkdir(parents=True,exist_ok=False);rows=[];output_error=0.
for item,bound in zip(cohort,f['rows']):
    assert item['low']==bound['low']
    assert sha(a.low_root/item['low'])==item['low_sha256'] and sha(a.normal_root/item['normal'])==item['normal_sha256']
    saved=torch.load(a.stage_a/bound['file'],map_location='cpu',weights_only=True)
    low=native_rgb(a.low_root/item['low']).cuda();normal=native_rgb(a.normal_root/item['normal']).cuda()
    model=CommonRegion2(torch.tensor(saved['gate']['active'],dtype=torch.bool)).cuda()
    for j,step in enumerate(saved['steps']):
        raw=saved['states'][j]
        with torch.no_grad():model.raw.copy_(raw.cuda())
        version=model.raw._version;output=model(low);error=float((output.detach().cpu()-saved['outputs'][j]).abs().max());output_error=max(output_error,error);assert error<=1e-6
        mse=(output.double()-normal.double()).square().mean();g_r,=torch.autograd.grad(mse,model.raw);g_r=g_r.detach().cpu();g_e=saved['g_e'][j]
        assert torch.isfinite(g_r).all() and torch.isfinite(g_e).all()
        assert model.raw._version==version and torch.equal(model.raw.detach().cpu(),raw) and model.raw.grad is None
        rows.append(dict(index=bound['index'],low=item['low'],step=step,selected=step==bound['selected_step'],prior_psnr_loss=item['low'] in loss,
            raw=raw.tolist(),g_e=g_e.tolist(),g_r=g_r.tolist(),masks={k:v.tolist() for k,v in saved['masks'].items()},groups={k:alignment(g_e,g_r,v) for k,v in saved['masks'].items()}))
    print(f'{bound["index"]+1}/100 reference attribution',flush=True)
selected=[r for r in rows if r['selected']];selected_loss=[r for r in selected if r['prior_psnr_loss']];nonloss=[r for r in selected if not r['prior_psnr_loss']]
assert len(rows)==f['audited_states'] and len(selected)==100 and len(selected_loss)==29 and len(nonloss)==71
sets=dict(all_selected=selected,loss_selected=selected_loss,nonloss_selected=nonloss,**{'step_'+str(s):[r for r in rows if r['step']==s] for s in STEPS})
summary=dict(label=LABEL,audited_states=len(rows),groups='active raw coordinates only: legacy channels0:2, gain channel2, total channels0:3',cosine_convention='T029 float64 active dot/norms; <=1e-12 norm is degenerate, cosine null; positive fraction among nondegenerate',aggregates={k:aggregate(v) for k,v in sets.items()},sign_patterns={k:patterns(v) for k,v in sets.items()})
summary['classification']=classify(summary['aggregates']['loss_selected'])
write(a.out/'states.json',rows);write(a.out/'summary.json',summary)
for row in f['rows']:assert sha(a.stage_a/row['file'])==row['sha256']
assert len([r for r in opened if r['kind']=='normal'])==100 and all(r['utc']>f['completed_utc'] for r in opened)
write(a.out/'receipt.json',dict(label=LABEL,started_utc=began,completed_utc=utc(),stage_a_freeze_sha256=sha(a.stage_a/'freeze.json'),stage_a_completed_utc=f['completed_utc'],prior_metrics_sha256=PRIOR_METRICS,
    opened_images=opened,audited_states=len(rows),all_finite=True,raw_unchanged_during_gradients=True,stage_a_all_file_hashes_unchanged=True,max_stage_a_output_abs_error=output_error,
    optimizer_updates=0,selection_changes=0,new_cohorts=0,official_test_access=False,seconds=time.perf_counter()-start,files={n:sha(a.out/n) for n in ['states.json','summary.json']}))
print(json.dumps(summary),flush=True)
