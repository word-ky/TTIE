"""One read-only audit of 4100 accepted states, plus deterministic sample verification."""
import time
import platform
from common import *
from reference_gradient import reference_gradient
from ttie.clip_signal import FrozenCLIP
from ttie.learned_prototypes import Prototypes
from ttie.semantic_ttt import FixedObjective,SemanticScorer
from ttie.energy_model import load_energy,features
from ttie.gamma_range_ttt import evaluate_energy

p=argparse.ArgumentParser(description=LABEL)
for k in ['preflight','split','low-root','normal-root','out']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();setup();start=time.perf_counter()
pre=json.loads((a.preflight/'preflight.json').read_bytes());assert sha(a.split)==SPLIT_SHA
assert sha(a.preflight/'bound_states.pt')==pre['bound_states_sha256']
states=torch.load(a.preflight/'bound_states.pt',weights_only=True,map_location='cpu')
split=json.loads(a.split.read_bytes())['selected'];accepted=Path(pre['accepted']);assets=pre['assets']
for v in assets.values():assert sha(v['path'])==v['sha256']
receipt=json.loads(Path(assets['gate']['path']).read_bytes())
encoder=FrozenCLIP.from_checkpoint(assets['clip']['path'],'cuda:0')
saved=torch.load(assets['prototypes']['path'],map_location='cuda:0',weights_only=True)
scorer=SemanticScorer(encoder,Prototypes(saved['raw']));head=load_energy(assets['energy']['path']).cuda()
def parameter_hashes():
    return {prefix+'.'+n:thash(t) for prefix,m in [('scorer',scorer),('head',head)] for n,t in m.state_dict().items()}
initial_hashes=parameter_hashes();rows=[];samples=[];gradients=[];image_times=[]
opened=[];normal_allowed={str((a.normal_root/r['normal']).resolve()) for r in split}
low_allowed={str((a.low_root/r['low']).resolve()) for r in split};original=Image.open
def paired_only(path,*args,**kwargs):
    name=str(Path(path).resolve());assert name in normal_allowed|low_allowed
    opened.append(dict(path=name,utc=utc(),kind='normal' if name in normal_allowed else 'low'))
    return original(path,*args,**kwargs)
Image.open=paired_only
a.out.mkdir(parents=True,exist_ok=False)
max_original_gradient_error=0.;max_feature_error=0.;max_sample_error=0.
for i,r in enumerate(split):
    torch.cuda.synchronize();image_start=time.perf_counter()
    low=native(a.low_root/r['low']);decision=json.loads((accepted/f'{i:03d}'/'decision.json').read_bytes())
    for n,h in pre['rows'][i]['files'].items():assert sha(accepted/f'{i:03d}'/n)==h
    objective=FixedObjective(scorer,low,receipt)
    for key,attr in [('scores','original_scores'),('active','active'),('winner','winner'),('evidence','evidence')]:
        assert torch.equal(getattr(objective,attr).cpu(),torch.tensor(decision['gate'][key],dtype=getattr(objective,attr).dtype))
    model,box=model_for(decision,low);mask=model.active.unsqueeze(0).unsqueeze(0).expand_as(model.raw).flatten()
    assert sha(a.normal_root/r['normal'])==r['normal_sha256']
    reference=native(a.normal_root/r['normal'])
    original_t=torch.load(accepted/f'{i:03d}'/'trajectory.pt',weights_only=True,map_location='cpu')
    pair_gradients=[]
    for step,state in enumerate(states[i]):
        assert thash(state)==pre['rows'][i]['state_sha256'][step]
        with torch.no_grad():model.raw.copy_(state.cuda())
        version=model.raw._version
        energy,output,scores,grid,f=evaluate_energy(model,low,objective,head)
        assert thash(output)==pre['rows'][i]['output_sha256'][step]
        assert torch.isfinite(output).all() and torch.isfinite(energy)
        max_feature_error=max(max_feature_error,float((f.detach().cpu()-original_t['features'][step]).abs().max()))
        g_e,=torch.autograd.grad(energy,model.raw)
        # Reference enters only this separate diagnostic call, after the low-only graph is consumed.
        g_r=reference_gradient(model,low,reference)
        assert torch.isfinite(g_e).all() and torch.isfinite(g_r).all()
        assert model.raw._version==version and torch.equal(model.raw.detach().cpu(),state) and model.raw.grad is None
        if step<40:
            old=torch.tensor(decision['diagnostics']['gradient_vectors'][step])
            error=float((g_e.cpu()-old).abs().max());max_original_gradient_error=max(max_original_gradient_error,error)
            torch.testing.assert_close(g_e.cpu(),old,rtol=1e-5,atol=1e-7)
        row=dict(index=i,image=r['low'],step=step,frozen_selected=step==pre['rows'][i]['selected_step'],**alignment(g_e,g_r,mask))
        rows.append(row);pair_gradients.append(torch.stack([g_e.detach().cpu(),g_r.detach().cpu()]))
        # Predeclared 30 samples: indices0,10,...90 x steps0,20,40. Fresh leaf, independent forward/backward.
        if i%10==0 and step in [0,20,40]:
            raw=state.cuda().clone().requires_grad_()
            y=torch.func.functional_call(model,{'raw':raw},(low,))
            from ttie.isp import physical_parameters
            physical=physical_parameters(torch.cat((raw,raw.new_zeros(1,4,2,2)),dim=1))[:,:2]
            e=head(features(objective,scorer(y),physical)).squeeze();e.backward()
            independent_e=raw.grad.clone();raw.grad=None
            y=torch.func.functional_call(model,{'raw':raw},(low,))
            ((y.double()-reference.double())**2).sum().div(y.numel()).backward()
            independent_r=raw.grad.clone()
            torch.testing.assert_close(independent_e,g_e,rtol=1e-5,atol=1e-7)
            torch.testing.assert_close(independent_r,g_r,rtol=1e-5,atol=1e-9)
            error=max(float((independent_e-g_e).abs().max()),float((independent_r-g_r).abs().max()))
            max_sample_error=max(max_sample_error,error)
            samples.append(dict(index=i,step=step,max_gradient_abs_error=error,
                independent_energy=independent_e.cpu().tolist(),independent_reference=independent_r.cpu().tolist(),
                **alignment(independent_e,independent_r,mask)))
            assert torch.equal(model.raw.detach().cpu(),state)
    gradients.append(torch.stack(pair_gradients));torch.cuda.synchronize();image_times.append(time.perf_counter()-image_start)
    print(f'{i+1}/100: 41 frozen states audited',flush=True)
assert len(rows)==4100 and len(samples)==30 and parameter_hashes()==initial_hashes
for v in assets.values():assert sha(v['path'])==v['sha256']
assert sha(a.preflight/'bound_states.pt')==pre['bound_states_sha256']
assert all(p.grad is None for m in [head,scorer] for p in m.parameters())
summary=summarize(rows)
write(a.out/'states.json',rows);torch.save(torch.stack(gradients),a.out/'gradients.pt')
write(a.out/'independent_samples.json',samples)
write(a.out/'summary.json',dict(all_states=summary,classification=classify(summary),
    frozen_selected=summarize([r for r in rows if r['frozen_selected']]),
    by_step={str(s):summarize([r for r in rows if r['step']==s]) for s in range(41)}))
write(a.out/'receipt.json',dict(label=LABEL,completed_utc=utc(),preflight_sha256=sha(a.preflight/'preflight.json'),
    preflight_completed_utc=pre['completed_utc'],opened_images=opened,states=4100,optimizer_updates=0,
    reference_selection_decisions=0,all_finite=True,raw_unchanged_during_gradients=True,model_parameter_hashes=initial_hashes,
    model_unchanged=True,checkpoint_unchanged=True,accepted_gradient_comparisons=4000,
    max_accepted_gradient_abs_error=max_original_gradient_error,max_feature_abs_error=max_feature_error,
    independent_samples=30,max_independent_gradient_abs_error=max_sample_error,seconds=time.perf_counter()-start,
    per_image_seconds=image_times,gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
    python=platform.python_version(),seed=7,tf32=False,raw_dtype='float32',reference_mse_accumulation='float64',
    files={n:sha(a.out/n) for n in ['states.json','gradients.pt','independent_samples.json','summary.json']}))
print(json.dumps(summary),classify(summary),flush=True)
