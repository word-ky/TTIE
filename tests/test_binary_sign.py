import copy
import hashlib
import json
import torch
from ttie.binary_sign import subset,fit,predict,axis_inputs

def test_heldout_poison_preserves_training_artifacts_and_prediction_hashes():
    torch.manual_seed(11);z=torch.randn(120,2,84)
    rows=[dict(row_index=i,bx=(.5,.4,.6)[i%3],by=(.6,.5,.4)[i%3]) for i in range(120)]
    changed=copy.deepcopy(rows)
    for r in changed[96:]:r.update(bx={'invalid':'heldout'},by=None)
    for ai,a in enumerate(('x','y')):
        first=subset(json.dumps(rows,indent=2).encode(),list(range(96)),a)
        second=subset(json.dumps(changed,indent=2).encode(),list(range(96)),a);assert first==second
        norm=dict(mean=z[:96,ai].double().mean(0).float().tolist(),scale=z[:96,ai].double().std(0,unbiased=False).float().tolist())
        h1,t1=fit(z[:,ai],*first,norm);h2,t2=fit(z[:,ai],*second,norm)
        artifact_hash=lambda h,t:hashlib.sha256(b''.join(v.numpy().tobytes() for v in h.state_dict().values())+json.dumps(t).encode()).hexdigest()
        assert artifact_hash(h1,t1)==artifact_hash(h2,t2)
        need=[i%2==0 for i in range(24)]
        p1=predict(h1,z[96:,ai],need);p2=predict(h2,z[96:,ai],need)
        assert hashlib.sha256(json.dumps(p1).encode()).hexdigest()==hashlib.sha256(json.dumps(p2).encode()).hexdigest()
        assert [v!=.5 for v in p1[2]]==need

def test_features_and_lower_first_tie():
    torch.manual_seed(17);f=torch.randn(120,5,28)
    z=axis_inputs([dict(row_index=i,features=v.tolist()) for i,v in enumerate(f)])
    assert torch.equal(z[:,0],torch.cat([f[:,0],f[:,1]-f[:,0],f[:,2]-f[:,0]],1))
    assert torch.equal(z[:,1],torch.cat([f[:,0],f[:,3]-f[:,0],f[:,4]-f[:,0]],1))
    class Tied(torch.nn.Module):
        def forward(self,x):return torch.zeros(len(x),2)
    assert predict(Tied(),z[:2,0],[False,True])==([[0.,0.],[0.,0.]],[0,0],[.5,.4])
