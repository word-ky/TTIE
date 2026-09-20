"""Independent post-freeze state rendering and objective-progress selection audit."""
import json,math
from pathlib import Path
from types import SimpleNamespace
import torch
import numpy as np
from ttie import gamma_range_ttt,common_gain_ttt
from ttie.common_gain import CommonRegion2
from research_log.T062A.core import losses
from research_log.T063D.cohort_verify import verify as verify_cohort
from research_log.T063D.infer import HERE,sha,tensor_hash,LOWROOT
from ttie.lolv2_gamma_core import native_rgb


def verify(out):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cohort=verify_cohort(HERE);f=json.loads((out/'freeze.json').read_bytes());config=json.loads((out/'config.json').read_bytes())
    for n,h in config['source_binding'].items():assert sha(n)==h,n
    for v in config['assets']['files'].values():assert sha(v['path'])==v['sha256']
    assert config['rho']==.9857470621423519 and config['T063_updates']==27 and config['control_updates']==40
    assert config['selector_sha256']==sha(HERE/'method.json') and config['cohort_sha256']==cohort['manifest_sha256']
    assert sha(HERE/'evaluation_binding.json')==json.loads((HERE/'method.json').read_bytes())['evaluation_binding_sha256']
    for n,h in json.loads((HERE/'evaluation_binding.json').read_bytes()).items():assert sha(n)==h,n
    assert f['opened_low_paths']==[str((LOWROOT/r['low']).resolve()) for r in f['rows']]
    for i,r in enumerate(f['rows']):
        d=out/f'{i:03d}';saved=torch.load(d/'outputs.pt',weights_only=True,map_location='cpu');x=native_rgb(LOWROOT/r['low']).cuda()
        assert torch.equal(x.cpu(),saved['identity']) and sha(LOWROOT/r['low'])==r['low_sha256']
        t=torch.load(d/'T063_trace.pt',weights_only=True,map_location='cpu');assert len(t['gradients'])==27 and len(t['states'])==28 and torch.count_nonzero(t['states'][0])==0
        model=CommonRegion2(t['active']).to(x).eval().requires_grad_(False);values=[];hashes=[]
        with torch.no_grad():
            for k,state in enumerate(t['states']):
                model.raw.copy_(state.to(x));y=model(x);parts=losses(x,y);values.append(float(parts@parts.new_tensor([1.,10.,5.])))
                np.testing.assert_allclose(parts.cpu().numpy(),t['components'][k].numpy(),rtol=0,atol=1e-7);hashes.append(tensor_hash(y.cpu()))
        np.testing.assert_allclose(values,t['values'],rtol=0,atol=1e-7)
        best=min(values);delta=values[0]-best;chosen=0
        if delta>1e-12:
            for k,v in enumerate(values):
                if math.isfinite(v) and v<=values[0]-.9857470621423519*delta:chosen=k;break
        assert chosen==t['selected_step']==r['methods']['T063']['selected_step']
        assert hashes[chosen]==r['methods']['T063']['output_hash']
        assert best==t['objective_best'] and delta==t['objective_reduction']
        q=[min(1.,max(0.,(values[0]-v)/max(delta,1e-12))) for v in values]
        np.testing.assert_array_equal(q,t['normalized_progress'])
        for n,module in [('T026',gamma_range_ttt),('T036',common_gain_ttt)]:
            record=torch.load(d/(n+'_trace.pt'),weights_only=True,map_location='cpu');trace=record['trace'];decision=record['decision'];active=torch.tensor(trace['gate']['active'])
            assert trace['diagnostics']['steps']==(40 if active.any() else 0)
            assert decision['selected_step']==min(range(len(decision['scores'])),key=lambda k:(decision['scores'][k],k))
            assert decision['scores']==trace['diagnostics']['loss_trajectory'] and torch.count_nonzero(trace['states'][0])==0
            model=module.make_model(SimpleNamespace(active=active),x,'region2').eval().requires_grad_(False)
            with torch.no_grad():model.raw.copy_(trace['states'][decision['selected_step']].to(x));image=model(x).cpu()
            assert torch.equal(image,saved[n]) and tensor_hash(image)==r['methods'][n]['output_hash']
        print('verified-states',i+1,flush=True)
    return dict(cohort=cohort,T063_states=2800,control_selected_states=200,objective_progress_choices=100)
