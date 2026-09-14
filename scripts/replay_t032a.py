"""Independent SciPy distances and NumPy prefix-selection replay, no image decode."""
import argparse,json,hashlib,math
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.distance import cdist

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    for k in ['audit','radius','spec']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--compact',action='store_true');a=p.parse_args();torch.set_num_threads(1)
    f=json.loads((a.audit/'freeze.json').read_bytes());rf=json.loads((a.radius/'freeze.json').read_bytes());spec=json.loads(a.spec.read_bytes())
    assert sha(a.radius/'freeze.json')==f['radius_freeze_sha256']==spec['radius_freeze_sha256']
    assert sha(a.spec)==f['spec_sha256'] and spec['r_support']==rf['r_support']
    assert sha(a.radius/'source.pt')==rf['source_sha256']
    source=torch.load(a.radius/'source.pt',weights_only=True,map_location='cpu')
    mean=source['x_mean'].double().numpy();scale=source['x_scale'].double().numpy();z=(source['source_features'].double().numpy()-mean)/scale
    rows=[];error=0.;states=0
    for r in f['rows']:
        d=a.audit/f'{r["index"]:03d}'
        for n,h in r['files'].items():
            if not a.compact or n in ['trajectory.pt','decision.json']:assert sha(d/n)==h['sha256']
        t=torch.load(d/'trajectory.pt',weights_only=True,map_location='cpu');decision=json.loads((d/'decision.json').read_bytes())
        q=(t['features'].double().numpy()-mean)/scale;values=cdist(q,z).min(1)/math.sqrt(28)
        error=max(error,float(np.abs(values-t['distances'].numpy()).max()));assert error<=1e-9
        energies=np.asarray(decision['original']['scores']);original=int(np.argmin(energies));exits=np.flatnonzero(values>rf['r_support'])
        exit_step=int(exits[0]) if len(exits) else None;cutoff=len(values)-1 if exit_step is None else max(0,exit_step-1)
        chosen=int(np.argmin(energies[:cutoff+1]))
        expected=dict(original_step=original,selected_step=chosen,cutoff=cutoff,exit_step=exit_step)
        assert expected==decision['guarded'];assert original==decision['original']['selected_step']
        assert (r['original_step'],r['selected_step'],r['cutoff'],r['crossing'])==(original,chosen,cutoff,exit_step)
        if not a.compact:
            for name,step in [('original',original),('guarded',chosen)]:
                saved=torch.load(d/(name+'.pt'),weights_only=True,map_location='cpu')
                assert torch.equal(saved['raw'],t['states'][step]) and torch.equal(saved['grid'],t['grids'][step])
                assert torch.isfinite(saved['image']).all()
        states+=len(values);rows.append(dict(index=r['index'],**expected))
    result=dict(status='PASS',count=len(rows),states=states,decisions=2*len(rows),distance_max_abs_error=error,rows=rows,mode='compact' if a.compact else 'full')
    (a.audit/('compact_replay.json' if a.compact else 'independent_replay.json')).write_text(json.dumps(result,indent=2)+'\n')
    print('PASS independent states',states,'decisions',2*len(rows),'max error',error,flush=True)

if __name__=='__main__':main()
