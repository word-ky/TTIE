"""Independent replay in a five-file bundle containing no targets or references."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch import nn

parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True);parser.add_argument('--receipt-sha',required=True)
args=parser.parse_args();root=args.bundle;torch.set_num_threads(1)
opened=[]
def read_bytes(name):
    opened.append(name)
    return (root/name).read_bytes()
raw=read_bytes('selector_frozen.json');assert hashlib.sha256(raw).hexdigest()==args.receipt_sha
receipt=json.loads(raw)
feature_raw=read_bytes('replay_features.json');expected_raw=read_bytes('replay_expected.json')
assert hashlib.sha256(feature_raw).hexdigest()==receipt['files_sha256']['replay_features.json']
assert hashlib.sha256(expected_raw).hexdigest()==receipt['files_sha256']['replay_expected.json']
features=json.loads(feature_raw);expected=json.loads(expected_raw)
assert set(features)=={'center','x_lower','x_upper','y_lower','y_upper'}
f={k:torch.tensor(v,dtype=torch.float32) for k,v in features.items()}
assert all(t.shape==(120,28) for t in f.values())
classes=[.5,.4,.6];results={};norms={}
for axis in ('x','y'):
    z=torch.cat([f['center'],f[axis+'_lower']-f['center'],f[axis+'_upper']-f['center']],dim=1)
    mean=z.double().mean(0).float();scale=z.double().std(0,unbiased=False).clamp_min(1e-12).float()
    filename='head_'+axis+'.pt';head_raw=read_bytes(filename)
    assert hashlib.sha256(head_raw).hexdigest()==receipt['files_sha256'][filename]
    import io
    checkpoint=torch.load(io.BytesIO(head_raw),map_location='cpu',weights_only=True)
    state=checkpoint['state_dict'];assert checkpoint['recipe']==receipt['recipe']
    assert torch.equal(state['x_mean'],mean) and torch.equal(state['x_scale'],scale)
    assert receipt['normalization'][axis]==dict(mean=mean.tolist(),scale=scale.tolist())
    net=nn.Sequential(nn.Linear(84,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,3)).eval()
    net.load_state_dict({k[4:]:v for k,v in state.items() if k.startswith('net.')})
    with torch.no_grad():logits=net((z-mean)/scale)
    results[axis]=(logits.tolist(),logits.argmax(1).tolist());norms[axis]=True
hard=[(x,y) for x in (.4,.5,.6) for y in (.4,.5,.6)];decisions=[]
for i in range(120):
    cx=results['x'][1][i];cy=results['y'][1][i];bx=classes[cx];by=classes[cy];index=hard.index((bx,by))
    decisions.append(dict(x_logits=results['x'][0][i],y_logits=results['y'][0][i],x_class=cx,y_class=cy,bx=bx,by=by,score_index=index,hard_index=3*index))
assert decisions==expected
assert set(opened)=={'selector_frozen.json','replay_features.json','replay_expected.json','head_x.pt','head_y.pt'}
print(json.dumps(dict(passed=True,exact_logits_classes_decisions=120,normalization_all_120_exact=norms,
    opened_artifacts=opened,target_or_reference_artifact_reads=False,ttie_imports=False,training=False)))
