"""Hash and reconstruct all accepted states before any reference image opens."""
from common import *
p=argparse.ArgumentParser()
for k in ['accepted','split','low-root','out']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();setup();assert sha(a.split)==SPLIT_SHA
split=json.loads(a.split.read_bytes())['selected'];assert len(split)==100
freeze=json.loads((a.accepted/'freeze.json').read_bytes());config=json.loads((a.accepted/'config.json').read_bytes())
assets=config['assets']['files']
for v in assets.values():assert sha(v['path'])==v['sha256']
assert assets['energy']['sha256']==ENERGY_SHA
bindings=json.loads(Path('research_log/T029A_alignment/source_binding.json').read_bytes())
for name,h in bindings['files'].items():assert sha(name)==h['sha256']
opened=[];allowed={str((a.low_root/r['low']).resolve()) for r in split};original=Image.open
def low_only(path,*args,**kwargs):
    name=str(Path(path).resolve());assert name in allowed;opened.append(name)
    return original(path,*args,**kwargs)
Image.open=low_only
a.out.mkdir(parents=True,exist_ok=False);rows=[];states=[]
for i,r in enumerate(split):
    d=a.accepted/f'{i:03d}';old=freeze['rows'][i];assert old['low']==r['low']
    hashes={n:sha(d/n) for n in ['output.pt','decision.json','trajectory.pt']}
    assert all(h==old['files'][n]['sha256'] for n,h in hashes.items())
    assert sha(a.low_root/r['low'])==r['low_sha256']
    decision=json.loads((d/'decision.json').read_bytes());low=native(a.low_root/r['low'])
    model,box=model_for(decision,low)
    t=torch.load(d/'trajectory.pt',map_location='cpu',weights_only=True)
    saved=torch.load(d/'output.pt',map_location='cpu',weights_only=True)
    assert t['states'].shape==(41,1,2,2,2);selected=decision['selection']['selected_step']
    assert torch.equal(t['states'][selected],saved['raw'])
    output_hashes=[]
    with torch.no_grad():
        for step,state in enumerate(t['states']):
            model.raw.copy_(state.cuda());output=model(low);grid=model.physical_grid()[:,:2]
            assert torch.isfinite(state).all() and torch.isfinite(output).all()
            assert torch.equal(grid.cpu(),t['grids'][step])
            assert ((grid>=box.lower-1e-6)&(grid<=box.upper+1e-6)).all()
            output_hashes.append(thash(output))
            if step==selected:assert torch.equal(output.cpu(),saved['image'])
    states.append(t['states'])
    rows.append(dict(index=i,low=r['low'],files=hashes,gate=decision['gate'],box=decision['diagnostics']['action_box'],
        selected_step=selected,state_sha256=[thash(s) for s in t['states']],output_sha256=output_hashes))
torch.save(torch.stack(states),a.out/'bound_states.pt')
assert len(opened)==100
write(a.out/'preflight.json',dict(label=LABEL,completed_utc=utc(),states=4100,pairs=100,rows=rows,assets=assets,
    accepted=str(a.accepted),accepted_source=bindings['accepted_source'],source_binding=bindings,
    accepted_files={n:sha(a.accepted/n) for n in ['freeze.json','config.json','metrics.csv']},
    split_sha256=sha(a.split),bound_states_sha256=sha(a.out/'bound_states.pt'),
    exact_all_grids=True,exact_selected_outputs=True,all_outputs_finite=True,normal_images_opened=0,low_images_opened=opened))
print('PREFLIGHT PASS: 4100 states and output hashes; 100 exact selected reconstructions; no reference decode',flush=True)
