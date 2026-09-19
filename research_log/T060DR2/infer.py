"""Exactly two literal trajectories; all outputs globally frozen before references."""
import argparse,time
from research_log.T060DR2.preflight import *
from research_log.T060CR1.core import trajectory as hybrid
from ttie.common_gain_ttt import trajectory as original

PREFLIGHT=ROOT/'research_log/T060DR2_preflight/preflight.json'

def episode(name,low,scorer,receipt,h014,hE):
    if name=='B':return hybrid(low,scorer,receipt,h014,hE)
    result,t,d=original(low,scorer,receipt,h014,basis='region2',max_steps=40)
    return result,t,d,[dict(g014=torch.tensor(g),hybrid=torch.tensor(g)) for g in t['diagnostics']['gradient_vectors']]

def main(out):
    setup();start=utc();out.mkdir(parents=True,exist_ok=False)
    pre=json.loads(PREFLIGHT.read_bytes());assert pre['status']=='PASS' and len(pre['checks'])==60
    assert sha(PREFLIGHT)==json.loads((HERE/'run_binding.json').read_bytes())['preflight_sha256']
    s=json.loads(Path('research_log/T060B/selection.json').read_bytes());binding=json.loads((HERE/'source_binding.json').read_bytes())
    # Only config/model/low inputs; historical prediction records are not trajectory inputs.
    inputs={n:h for n,h in pre['inputs'].items() if not n.startswith(str(PRED)+'/')};inputs[str(PREFLIGHT)]=sha(PREFLIGHT)
    firewall(set(inputs)|{str(Path(n).resolve()) for n in binding}|{str((HERE/'source_binding.json').resolve())},out)
    validate(binding);validate(inputs);scorer,receipt,h014,hE=models()
    before={n:{k:thash(v) for k,v in h.state_dict().items()} for n,h in [('014',h014),('E',hE)]};rows=[]
    for i,r in enumerate(s['anchors']):
        low=torch.load(Path(r['directory'])/'bank_images.pt',weights_only=True,map_location='cpu')[0].cuda();assert thash(low)==r['state_before']['y0'];methods={}
        for name in ['A','B']:
            begin=time.perf_counter();result,t,d,trace=episode(name,low,scorer,receipt,h014,hE)
            assert len(t['states'])==41 and len(trace)==40 and t['diagnostics']['parameter_count']==12
            assert torch.count_nonzero(t['states'][0])==0
            assert t['gate']['active']==r['gate']['active'] and t['gate']['winner']==r['gate']['winner']
            assert np.max(abs(np.array(t['gate']['scores'])-np.array(r['gate']['scores'])))<=1e-5
            assert all(torch.isfinite(t[k]).all() for k in ['images','states','scores','grids','features'])
            folder=out/f'{i:03d}'/name;folder.mkdir(parents=True)
            save_tensor(folder/'images.pt',t['images']);save_tensor(folder/'trajectory.pt',{k:t[k] for k in ['states','scores','grids','features']});save_tensor(folder/'gradients.pt',trace)
            save_tensor(folder/'output.pt',{k:result[k].detach().cpu().clone() for k in ['image','raw','grid']})
            atomic_json(folder/'decision.json',dict(selection=d,gate=t['gate'],diagnostics=t['diagnostics']))
            methods[name]=dict(selected_step=d['selected_step'],states=41,updates=40,seconds=time.perf_counter()-begin,files={p.name:sha(p) for p in folder.iterdir()},persisted_utc=utc())
            del result,t,trace
        rows.append(dict(order=i,index=r['index'],bank_index=r['bank_index'],image_id=r['image_id'],low_hash=thash(low),methods=methods))
        atomic_json(out/'progress.json',rows);print(json.dumps(dict(completed=i+1,seconds=sum(x['seconds'] for x in methods.values()))),flush=True)
    after={n:{k:thash(v) for k,v in h.state_dict().items()} for n,h in [('014',h014),('E',hE)]};assert before==after
    validate(binding);validate(inputs)
    atomic_json(out/'freeze.json',dict(started_utc=start,completed_utc=utc(),rows=rows,inputs=inputs,source_bindings=binding,preflight_sha256=sha(PREFLIGHT),head_state_before=before,head_state_after=after,normalizations={'014':h014.normalization(),'E':hE.normalization()},optimizer_steps=4800,source_clean_reads=0,reference_gradient_reads=0,metric_reads=0,target_domain_access=0,official_test_access=0,physical_gpu=1,gpu=torch.cuda.get_device_name()))
    print('ALL_120_TRAJECTORIES_FROZEN',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
