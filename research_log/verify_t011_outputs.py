"""One-shot audit of finalized T011 files; never reads clean references or updates a model."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def main():
    parser=argparse.ArgumentParser();parser.add_argument('audit',type=Path);args=parser.parse_args()
    root=args.audit;artifacts=json.loads((root/'artifact_manifest.json').read_text())
    config=json.loads((root/'config.json').read_text());methods=config['methods']
    final=json.loads((root/'final_checks.json').read_text())
    assert len(artifacts)==final['inputs']==240 and final['rows']==2160
    totals=dict(inputs=0,outputs=0,updates=0,raw_states=0,projected_updates=0,no_active_inputs=0,
                output_bytes=0,state_bytes=0,decision_bytes=0,inactive_region_exact_checks=0)
    for entry in artifacts:
        directory=root/entry['directory'];receipt=json.loads((directory/'label_free_receipt.json').read_text())
        for key,total in (('outputs','output_bytes'),('states','state_bytes'),('decisions','decision_bytes')):
            r=entry[key];path=directory/r['file']
            assert receipt[key]==r
            assert path.stat().st_size==r['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256']
            totals[total]+=r['bytes']
        pack=torch.load(directory/'outputs.pt',map_location='cpu',weights_only=True)
        states=torch.load(directory/'states.pt',map_location='cpu',weights_only=True)
        decisions=json.loads((directory/'decisions.json').read_text());active=torch.tensor(decisions['gate']['active'])
        assert list(pack)==methods and list(decisions['methods'])==methods
        source=pack['identity']['image'];h,w=source.shape[-2:]
        all_inactive=not bool(active.any());totals['no_active_inputs']+=int(all_inactive)
        for name,result in pack.items():
            diag=decisions['methods'][name];steps=diag['steps'];totals['updates']+=steps
            for tensor in result.values():assert torch.isfinite(tensor).all()
            assert result['image'].shape==source.shape
            assert (result['image']>=0).all() and (result['image']<=1).all()
            assert torch.allclose(result['grid'],torch.tensor(diag['final_grid']),atol=0,rtol=0)
            if name in states:
                state=states[name];totals['raw_states']+=len(state['states'])
                assert torch.equal(state['states'][0],torch.zeros_like(result['raw']))
                assert torch.equal(state['final_raw'],result['raw'])
                assert torch.equal(state['states'][-1],result['raw'])
                if 'ttt' in name:assert len(state['states'])==steps+1
            if 'ttt' in name:
                assert len(diag['gradient_vectors'])==len(diag['gradient_norms'])==steps
                assert len(diag['loss_trajectory'])==steps+1
                for vector,norm in zip(diag['gradient_vectors'],diag['gradient_norms']):
                    vector=torch.tensor(vector);assert torch.isfinite(vector).all()
                    assert abs(float(vector.norm())-norm)<=1e-5*max(1.,norm)
            if all_inactive:
                assert steps==0 and torch.equal(result['image'],source) and torch.equal(result['raw'],torch.zeros_like(result['raw']))
            if name.endswith('_1step'):assert steps==(0 if all_inactive else 1)
            if 'projected' in name:
                lower=torch.tensor(diag['action_box']['lower']);upper=torch.tensor(diag['action_box']['upper'])
                assert (result['grid']>=lower-1e-6).all() and (result['grid']<=upper+1e-6).all()
                for i,projection in enumerate(diag['projections']):
                    grid=torch.tensor(projection['post_grid']);raw=torch.tensor(projection['post_raw'])
                    assert (grid>=lower-1e-6).all() and (grid<=upper+1e-6).all()
                    assert torch.equal(raw,states[name]['states'][i+1])
                    assert torch.isfinite(torch.tensor(projection['pre_raw'])).all()
                    assert torch.isfinite(torch.tensor(projection['pre_grid'])).all()
                if 'ttt' in name:
                    assert len(diag['projections'])==steps;totals['projected_updates']+=steps
                if name.startswith('region2'):
                    for i,is_active in enumerate(active):
                        if is_active:continue
                        y,x=divmod(i,2);ys=slice(0,h//2) if y==0 else slice(h//2,h)
                        xs=slice(0,w//2) if x==0 else slice(w//2,w)
                        assert torch.equal(result['image'][...,ys,xs],source[...,ys,xs])
                        assert torch.equal(result['raw'][:,:,y,x],torch.zeros(1,2))
                        totals['inactive_region_exact_checks']+=1
            if name=='region2_discrete_projected':
                for search in diag['search']:
                    y,x=search['node'];i=2*y+x;channel=search['channel']
                    expected=([0.] if channel==0 else [1.]) if not active[i] else ([.8,1.,1.25] if channel==1 else
                        [0.,.25,.5] if decisions['gate']['winner'][i]==0 else [-.5,-.25,0.])
                    assert search['candidates']==expected and search['chosen'] in expected
        totals['inputs']+=1;totals['outputs']+=len(pack)
    report=dict(task='T011',all_output_state_decision_hashes_verified=True,finite_and_bounds=True,
        identity_reset_and_no_active_exact=True,inactive_projected_regions_bitwise_identity=True,
        gradient_and_projection_trace_lengths=True,one_step_counts_exact=True,discrete_candidates_legal=True,**totals)
    (root/'output_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
