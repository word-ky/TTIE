import copy
import hashlib
import json
import torch
from ttie.deadband_probe import training_targets
from ttie.direction_probe import fit_axis_fold
from ttie.nonspatial_oof import feature_inputs
from ttie.local_geometry import write,sha


def test_cross_scatter_preserves_exact_axis_features(tmp_path):
    torch.manual_seed(13); f=torch.randn(120,5,28)
    write(tmp_path/'features.json',[dict(row_index=i,features=v.tolist()) for i,v in enumerate(f)])
    write(tmp_path/'features_frozen.json',dict(features_sha256=sha(tmp_path/'features.json')))
    z=feature_inputs(tmp_path)
    assert torch.equal(z[:,0],torch.cat([f[:,0],f[:,1]-f[:,0],f[:,2]-f[:,0]],1))
    assert torch.equal(z[:,1],torch.cat([f[:,0],f[:,3]-f[:,0],f[:,4]-f[:,0]],1))


def test_heldout_target_mutation_prediction_hash_isolation():
    torch.manual_seed(13);z=torch.randn(120,2,84)
    rows=[dict(row_index=i,bx=(.5,.4,.6)[i%3],by=(.5,.4,.6)[(i//3)%3]) for i in range(120)]
    fold=dict(train=list(range(96)),heldout=list(range(96,120)))
    mutated=copy.deepcopy(rows)
    for i in fold['heldout']:mutated[i].update(bx='forbidden heldout label',by=None)
    original=training_targets(json.dumps(rows,indent=2).encode(),fold['train'])
    changed=training_targets(json.dumps(mutated,indent=2).encode(),fold['train'])
    assert original==changed
    for axis in (0,1):
        first=fit_axis_fold(z,original,fold,axis)[2:]
        second=fit_axis_fold(z,changed,fold,axis)[2:]
        digest=lambda v:hashlib.sha256(json.dumps(v).encode()).hexdigest()
        assert digest(first)==digest(second)
