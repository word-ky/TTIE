"""Reference gradients only after all100 low-only probes are frozen."""
from core import *
import argparse,time
from PIL import Image
from scripts.run_t036a import initialize
from ttie.common_gain import CommonRegion2
from ttie.lolv2_gamma_core import native_rgb
p=argparse.ArgumentParser()
for k in ['stage-a','manifest','low-root','normal-root','out']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();initialize();start=time.perf_counter();began=utc()
f=json.loads((a.stage_a/'freeze.json').read_bytes());assert sha(a.manifest)==COHORT==f['cohort_sha256']
assert f['audited_states']==100 and f['legacy_step']==10 and f['prior_reference_gradient_files_opened']==f['normal_decodes']==f['prior_metric_files_opened']==f['prior_loss_id_files_opened']==f['optimizer_updates']==f['selection_changes']==0
for name,h in f['source_binding'].items():assert sha(name)==h
for row in f['rows']:assert sha(a.stage_a/row['file'])==row['sha256']
assert sha(a.stage_a/'gates.json')==f['gates_sha256']
cohort=json.loads(a.manifest.read_bytes())['selected'];baselines=json.loads(Path('research_log/T042A_baseline_bindings.json').read_bytes());baseline_opened=[]
for name,v in baselines.items():
    baseline_opened.append(dict(path=v['path'],utc=utc()))
    assert sha(v['path'])==v['sha256']
    v['groups']=json.loads(Path(v['path']).read_bytes())['by_gain']['1.75']
assert baselines['selected']['groups']['total']['positive_dot_fraction']==.37 and baselines['selected']['groups']['total']['cosine_median']==-.18584799268346364
opened=[];normal_allowed={str((a.normal_root/r['normal']).resolve()) for r in cohort};low_allowed={str((a.low_root/r['low']).resolve()) for r in cohort};original=Image.open
def paired(path,*args,**kwargs):
    name=str(Path(path).resolve());assert name in normal_allowed|low_allowed
    opened.append(dict(path=name,utc=utc(),kind='normal' if name in normal_allowed else 'low'));return original(path,*args,**kwargs)
Image.open=paired;a.out.mkdir(parents=True,exist_ok=False);rows=[]
for item,bound in zip(cohort,f['rows']):
    assert item['low']==bound['low']
    assert sha(a.low_root/item['low'])==item['low_sha256'] and sha(a.normal_root/item['normal'])==item['normal_sha256']
    saved=torch.load(a.stage_a/bound['file'],map_location='cpu',weights_only=True)
    low=native_rgb(a.low_root/item['low']).cuda();normal=native_rgb(a.normal_root/item['normal']).cuda()
    model=CommonRegion2(torch.tensor(saved['gate']['active'],dtype=torch.bool)).cuda()
    for j,gain in enumerate(GAINS):
        raw=saved['states'][j];assert torch.equal(raw[:,:2],saved['legacy']) and thash(raw[:,:2])==bound['legacy_sha256']
        with torch.no_grad():model.raw.copy_(raw.cuda())
        version=model.raw._version;output=model(low)
        assert thash(output)==saved['output_hashes'][j]==bound['output_hashes'][j] and torch.equal(output.detach().cpu(),saved['outputs'][j])
        loss=(output.double()-normal.double()).square().mean();g_r,=torch.autograd.grad(loss,model.raw);g_r=g_r.cpu();g_e=saved['g_e'][j]
        assert torch.isfinite(g_r).all() and torch.isfinite(g_e).all() and torch.isfinite(loss)
        assert model.raw._version==version and torch.equal(model.raw.detach().cpu(),raw) and model.raw.grad is None
        rows.append(dict(index=bound['index'],low=item['low'],legacy_step=bound['legacy_step'],legacy_sha256=bound['legacy_sha256'],gain=gain,raw=raw.tolist(),g_e=g_e.tolist(),g_r=g_r.tolist(),energy=saved['energies'][j],output_sha256=thash(output),masks={k:v.tolist() for k,v in saved['masks'].items()},groups={k:alignment(g_e,g_r,v) for k,v in saved['masks'].items()}))
    print(f'{bound["index"]+1}/100 reference gradients',flush=True)
assert len(rows)==100
summary=dict(label=LABEL,audited_states=100,legacy_step=10,fixed_gain=1.75,groups=aggregate(rows),baselines=baselines)
summary['step10_minus_baseline']={name:{group:{k:summary['groups'][group][k]-v['groups'][group][k] for k in ['positive_dot_fraction','cosine_median']} for group in ['legacy','gain','total']} for name,v in baselines.items()}
summary['classification']=classify(summary['groups']['total'])
write(a.out/'states.json',rows);write(a.out/'summary.json',summary)
for row in f['rows']:assert sha(a.stage_a/row['file'])==row['sha256']
assert len([r for r in opened if r['kind']=='normal'])==100 and all(r['utc']>f['completed_utc'] for r in opened)
write(a.out/'receipt.json',dict(label=LABEL,started_utc=began,completed_utc=utc(),stage_a_freeze_sha256=sha(a.stage_a/'freeze.json'),stage_a_completed_utc=f['completed_utc'],opened_images=opened,audited_states=100,output_hashes_exact=100,baseline_opened=baseline_opened,all_finite=True,raw_unchanged_during_gradients=True,stage_a_all_file_hashes_unchanged=True,optimizer_updates=0,selection_changes=0,prior_metric_files_opened=0,prior_loss_id_files_opened=0,new_cohorts=0,official_test_access=False,seconds=time.perf_counter()-start,files={n:sha(a.out/n) for n in ['states.json','summary.json']}))
print(json.dumps(summary),flush=True)
