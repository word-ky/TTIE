"""Check frozen low-only selections, fields and outputs before reference deployment."""
import argparse,json
from pathlib import Path
import torch
from scripts.run_t036a import sha,write,utc
from ttie.common_gain import physical_grid
p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);a=p.parse_args()
f=json.loads((a.audit/'freeze.json').read_bytes());assert len(f['rows'])==100 and f['normal_decodes']==0
counts={n:0 for n in ['baseline','common']};states={n:0 for n in counts};gains=[]
for r in f['rows']:
    gates=[]
    for name,record in r['methods'].items():
        d=a.audit/f'{r["index"]:03d}'/name
        for n,h in record['files'].items():assert sha(d/n)==h['sha256']
        dec=json.loads((d/'decision.json').read_bytes());gates.append(dec['gate'])
        t=torch.load(d/'trajectory.pt',weights_only=True,map_location='cpu');out=torch.load(d/'output.pt',weights_only=True,map_location='cpu')
        fields=torch.load(d/'fast_fields.pt',weights_only=True,map_location='cpu')['fields'];active=torch.tensor(dec['gate']['active']).reshape(2,2)
        budget=40 if active.any() else 0;assert record['updates']==budget and len(t['states'])==budget+1
        score=dec['selection']['scores'];index=min(range(len(score)),key=lambda i:(score[i],i)) if active.any() else 0
        assert index==record['selected_step']==dec['selection']['selected_step']
        assert torch.equal(out['raw'],t['states'][index]) and torch.equal(out['grid'],t['grids'][index])
        assert all(torch.isfinite(v).all() for v in [out['image'],t['states'],fields,t['scores'],t['features'],torch.tensor(score)])
        assert 0<=out['image'].min()<=out['image'].max()<=1
        raw=t['states'][:,0]
        if name=='baseline':raw=torch.cat([raw,raw.new_zeros(len(raw),1,2,2)],1)
        expected=physical_grid(raw)[:,[0,1,2]];assert torch.equal(expected,fields)
        lo=torch.tensor(dec['diagnostics']['action_box']['lower']);hi=torch.tensor(dec['diagnostics']['action_box']['upper'])
        assert (fields[:,:2]>=lo-1e-6).all() and (fields[:,:2]<=hi+1e-6).all()
        assert (fields[:,2]>=.5).all() and (fields[:,2]<=2).all()
        assert torch.equal(fields[:,2][:,~active],torch.ones_like(fields[:,2][:,~active]))
        counts[name]+=budget;states[name]+=len(raw)
    assert gates[0]==gates[1]
write(a.audit/'independent_replay.json',dict(status='PASS',completed_utc=utc(),count=100,methods=200,updates=counts,states=states,
    all800_artifact_hashes_verified=True,all_energy_minima_verified=True,all_fields_finite_bounded=True,inactive_gain_identity=True,identical_gates=True))
print('PASS200 frozen low-only episodes',counts,states,flush=True)
