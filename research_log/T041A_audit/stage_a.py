"""Low-only parity preflight and all learned gradients; freeze before references."""
from core import *
import argparse,time
from PIL import Image
from scripts.run_t036a import initialize,models
from ttie.lolv2_gamma_core import native_rgb,low_image_opener
from ttie.semantic_ttt import FixedObjective
from ttie.common_gain import CommonRegion2
from ttie.common_gain_ttt import evaluate_energy
p=argparse.ArgumentParser()
for k in ['accepted','manifest','assets','binding','low-root','out']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();initialize();start=time.perf_counter()
assert sha(a.manifest)==COHORT and sha(a.accepted/'freeze.json')==PRIOR_FREEZE
frozen=json.loads((a.accepted/'freeze.json').read_bytes());cohort=json.loads(a.manifest.read_bytes())['selected'];assets=json.loads(a.assets.read_bytes())
assert sha(a.accepted/'config.json')==frozen['config_sha256']
assert assets==json.loads((a.accepted/'config.json').read_bytes())['assets']
binding=json.loads(a.binding.read_bytes())
for name,h in binding.items():assert sha(name)==h,name
opened=[];allowed={str((a.low_root/r['low']).resolve()) for r in cohort};Image.open=low_image_opener(allowed,opened)
a.out.mkdir(parents=True,exist_ok=False);preflight=[]
# Every retained selected output is checked before the first gradient calculation.
for i,(item,old) in enumerate(zip(cohort,frozen['rows'])):
    assert item['low']==old['low'] and sha(a.low_root/item['low'])==item['low_sha256']
    source=a.accepted/f'{i:03d}'/'common'
    for n,h in old['methods']['common']['files'].items():assert sha(source/n)==h['sha256']
    d=json.loads((source/'decision.json').read_bytes());saved=torch.load(source/'output.pt',map_location='cpu',weights_only=True)
    t=torch.load(source/'trajectory.pt',map_location='cpu',weights_only=True);selected=d['selection']['selected_step']
    assert selected==old['methods']['common']['selected_step']==min(range(41),key=lambda s:d['selection']['scores'][s])
    low=native_rgb(a.low_root/item['low']).cuda();model=CommonRegion2(torch.tensor(d['gate']['active'],dtype=torch.bool)).cuda()
    with torch.no_grad():model.raw.copy_(t['states'][selected].cuda());output=model(low).cpu()
    error=float((output-saved['image']).abs().max());assert error<=1e-6
    preflight.append(dict(index=i,low=item['low'],selected_step=selected,selected_output_max_abs=error,bit_exact=torch.equal(output,saved['image']),selected_output_sha256=thash(output),prior_files=old['methods']['common']['files']))
assert len(preflight)==100
write(a.out/'parity.json',dict(completed_utc=utc(),normal_decodes=0,gradient_calls=0,rows=preflight))
scorer,gate,head=models(assets);head.eval().requires_grad_(False);scorer.eval().requires_grad_(False)
def parameter_hashes():return {prefix+'.'+n:thash(v) for prefix,m in [('scorer',scorer),('head',head)] for n,v in m.state_dict().items()}
initial=parameter_hashes();rows=[];gate_rows=[]
for i,(item,pre) in enumerate(zip(cohort,preflight)):
    low=native_rgb(a.low_root/item['low']).cuda();source=a.accepted/f'{i:03d}'/'common'
    d=json.loads((source/'decision.json').read_bytes());t=torch.load(source/'trajectory.pt',map_location='cpu',weights_only=True)
    objective=FixedObjective(scorer,low,gate)
    for key,attr in [('scores','original_scores'),('active','active'),('winner','winner'),('evidence','evidence')]:
        assert torch.equal(getattr(objective,attr).cpu(),torch.tensor(d['gate'][key],dtype=getattr(objective,attr).dtype))
    model=CommonRegion2(objective.active).cuda();groups=masks(objective.active.cpu())
    states=[];outputs=[];gradients=[];energies=[];features_saved=[]
    legacy=t['states'][pre['selected_step']][:,:2]
    for gain in GAINS:
        raw=probe_raw(legacy,objective.active.cpu(),gain)
        assert torch.equal(raw[:,:2],legacy)
        with torch.no_grad():model.raw.copy_(raw.cuda())
        version=model.raw._version;energy,output,_,_,features=evaluate_energy(model,low,objective,head)
        g_e,=torch.autograd.grad(energy,model.raw)
        assert all(torch.isfinite(v).all() for v in [raw,g_e,output,energy,features])
        assert model.raw._version==version and torch.equal(model.raw.detach().cpu(),raw) and model.raw.grad is None
        states.append(raw);outputs.append(output.detach().cpu());gradients.append(g_e.detach().cpu());energies.append(float(energy.detach()));features_saved.append(features.detach().cpu())
    file=a.out/f'{i:03d}.pt'
    output_hashes=[thash(v) for v in outputs]
    torch.save(dict(label=LABEL,gains=GAINS,states=torch.stack(states),outputs=torch.stack(outputs),g_e=torch.stack(gradients),features=torch.stack(features_saved),energies=energies,gate=d['gate'],masks=groups,selected_step=pre['selected_step'],legacy=legacy,output_hashes=output_hashes),file)
    rows.append(dict(index=i,low=item['low'],file=file.name,sha256=sha(file),gains=GAINS,selected_step=pre['selected_step'],output_hashes=output_hashes))
    gate_rows.append(dict(index=i,low=item['low'],gate=d['gate'],stage_a_file_sha256=sha(file)))
    print(f'{i+1}/100 low-only gradients frozen',flush=True)
assert parameter_hashes()==initial and all(p.grad is None for m in [scorer,head] for p in m.parameters())
for v in assets['files'].values():assert sha(v['path'])==v['sha256']
assert len(opened)==200 and sha(a.accepted/'freeze.json')==PRIOR_FREEZE
write(a.out/'gates.json',gate_rows)
write(a.out/'freeze.json',dict(label=LABEL,completed_utc=utc(),rows=rows,audited_states=200,fixed_gains=GAINS,gates_sha256=sha(a.out/'gates.json'),cohort_sha256=COHORT,prior_freeze_sha256=PRIOR_FREEZE,
    parity_sha256=sha(a.out/'parity.json'),parity_completed_utc=json.loads((a.out/'parity.json').read_bytes())['completed_utc'],
    source_binding=binding,assets=assets,model_parameter_hashes=initial,model_unchanged=True,all_finite=True,raw_unchanged_during_gradients=True,
    prior_metric_files_opened=0,prior_loss_id_files_opened=0,
    opened_lows=opened,normal_decodes=0,optimizer_updates=0,selection_changes=0,gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,seconds=time.perf_counter()-start))
print('STAGE A FROZEN',sha(a.out/'freeze.json'),flush=True)
