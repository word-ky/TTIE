"""Independent NumPy selector replay from frozen low-only artifacts. No image IO."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
def replay(energies,gradients,active):
    v=np.asarray(gradients,dtype=np.float64).reshape(len(energies),-1)[:,np.tile(active,2).astype(bool)]
    norm=np.sqrt((v*v).sum(1));original=int(np.argmin(energies));cutoff=len(energies)-1;crossing=None;fallback=None
    if len(energies)<=10 or norm[10]<=1e-12:fallback='degenerate_anchor'
    else:
        for s in range(11,len(energies)):
            if norm[s]<=1e-12:fallback='degenerate_comparison';cutoff=len(energies)-1;break
            if np.sum(v[s]*v[10])/(norm[s]*norm[10])<=0:crossing=s;cutoff=s-1;break
    selected=original if fallback else int(np.argmin(energies[:cutoff+1]))
    return dict(original_step=original,selected_step=selected,cutoff=cutoff,crossing=crossing,fallback=fallback)
def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);a=p.parse_args()
    freeze=json.loads((a.audit/'freeze.json').read_bytes());result=[]
    for r in freeze['rows']:
        d=a.audit/f'{r["index"]:03d}'
        for n,h in r['files'].items():assert hashlib.sha256((d/n).read_bytes()).hexdigest()==h['sha256']
        decision=json.loads((d/'decision.json').read_bytes());t=torch.load(d/'trajectory.pt',map_location='cpu',weights_only=True)
        rep=replay(decision['original']['scores'],t['gradients'].numpy(),decision['gate']['active'])
        for k,v in rep.items():assert decision['guarded'][k]==v
        for name,step in [('original.pt',rep['original_step']),('guarded.pt',rep['selected_step'])]:
            saved=torch.load(d/name,map_location='cpu',weights_only=True)
            assert torch.equal(saved['raw'],t['states'][step]) and torch.equal(saved['grid'],t['grids'][step])
        result.append(dict(index=r['index'],**rep))
    (a.audit/'independent_replay.json').write_text(json.dumps(dict(status='PASS',count=len(result),rows=result),indent=2)+'\n')
    print('Independent low-only NumPy replay PASS',len(result))
if __name__=='__main__':main()
