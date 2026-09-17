"""Range-aware readback and immutable shard0/combined coverage verification."""
import json,hashlib
from pathlib import Path
import torch
from research_log.T058AD.storage import atomic_json,save_chunk,reopen as reopen_shard0
from research_log.T058A_tangent.core import sha,thash

def reopen(out,expected,first=1024):
    out=Path(out);manifest=json.loads((out/'manifest.json').read_bytes());rows=manifest['rows'];order=[]
    assert len(rows)==expected and [x['index'] for x in rows]==list(range(first,first+expected))
    for item in manifest['chunks']:
        file=out/item['file'];assert sha(file)==item['sha256']
        saved=torch.load(file,map_location='cpu',weights_only=True);grad=saved['g_E']
        assert grad.dtype==torch.float32 and grad.shape==(len(saved['indices']),1,1,8,8) and torch.isfinite(grad).all()
        assert len(saved['indices'])==item['count'] and saved['indices']==list(range(item['first'],item['last']+1))
        for i,g in zip(saved['indices'],grad):
            assert thash(g)==rows[i-first]['gradient_sha256'];assert float(g.double().norm())==rows[i-first]['gradient_norm'];order.append(i)
    assert order==list(range(first,first+expected))
    return dict(rows=expected,first=first,last=first+expected-1,execution_order_exact=True,all_finite=True,all_gradient_hashes_match=True,all_norms_match=True,manifest_sha256=sha(out/'manifest.json'),chunks=len(manifest['chunks']))

def verify_shard0(run,archive):
    run=Path(run);archive=json.loads(Path(archive).read_bytes());files=archive['files']
    for n,h in files.items():assert sha(run/n)==h,n
    out=run/'artifacts/T058AD';marker=json.loads((out/'complete.json').read_bytes())
    assert marker['rows']==1024 and marker['range']==[0,1023]
    assert marker['classification']=='T058-AD Stage-A shard 0 frozen'
    assert sha(out/'manifest.json')==marker['manifest_sha256'] and sha(out/'receipt.json')==marker['receipt_sha256']
    result=reopen_shard0(out,1024)
    return dict(run=str(run),files=files,complete_sha256=sha(out/'complete.json'),receipt_sha256=sha(out/'receipt.json'),manifest_sha256=sha(out/'manifest.json'),reopen=result)

def coverage(old_rows,new_rows,banks):
    expected=[]
    for bi,entry in enumerate(banks):
        for j in range(entry['states']):expected.append(dict(index=len(expected),bank_index=bi,state_index=j,image_id=entry['image_id']))
    actual=[{k:r[k] for k in ['index','bank_index','state_index','image_id']} for r in old_rows+new_rows]
    assert actual==expected,'COMBINED_CANONICAL_COVERAGE_FAILURE'
    indices=[r['index'] for r in actual];assert len(set(indices))==len(indices)
    return dict(rows=len(actual),unique_indices=len(set(indices)),first=indices[0],last=indices[-1],no_gap=True,no_overlap=True,canonical_identities_match=True,order_exact=True,canonical_identity_sha256=hashlib.sha256(json.dumps(actual,sort_keys=True).encode()).hexdigest())
