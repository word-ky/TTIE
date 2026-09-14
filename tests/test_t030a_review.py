import hashlib
import json
import sys
import pytest
import torch
from scripts import evaluate_t030a,replay_t030a
from ttie.self_reversal import select

def write(path,value):path.write_text(json.dumps(value))

def test_changed_freeze_rejected_before_reference_access(tmp_path,monkeypatch):
    freeze=tmp_path/'freeze.json';write(freeze,dict(completed_utc='2026-01-01'))
    original_hash=hashlib.sha256(freeze.read_bytes()).hexdigest()
    write(tmp_path/'deployment.json',dict(started_utc='2026-01-02',freeze_sha256=original_hash))
    # A replacement keeps an apparently valid timestamp but has different bytes.
    write(freeze,dict(completed_utc='2026-01-01',replacement=True))
    write(tmp_path/'manifest.json',{})
    def forbidden(*args,**kwargs):raise AssertionError('reference access before binding check')
    monkeypatch.setattr(evaluate_t030a,'pixels',forbidden)
    argv=['evaluate','--audit',str(tmp_path),'--manifest',str(tmp_path/'manifest.json'),
          '--normal-root',str(tmp_path/'absent_normals'),'--deployment',str(tmp_path/'deployment.json')]
    monkeypatch.setattr(sys,'argv',argv)
    with pytest.raises(AssertionError,match='audit freeze does not match reference deployment'):evaluate_t030a.main()

def test_compact_replay_without_images_preserves_full_receipt(tmp_path,monkeypatch):
    d=tmp_path/'000';d.mkdir();g=torch.ones(41,1,2,2,2);g[15]=-1
    energies=list(reversed(range(41)));active=[True]*4;decision=select(energies,g,torch.tensor(active))
    write(d/'decision.json',dict(original=dict(scores=energies),guarded=decision,gate=dict(active=active)))
    torch.save(dict(gradients=g),d/'trajectory.pt')
    files={n:dict(sha256=hashlib.sha256((d/n).read_bytes()).hexdigest()) for n in ['decision.json','trajectory.pt']}
    files.update({'original.pt':dict(sha256='omitted'),'guarded.pt':dict(sha256='omitted')})
    write(tmp_path/'freeze.json',dict(rows=[dict(index=0,files=files)]))
    previous=b'original full replay receipt';(tmp_path/'independent_replay.json').write_bytes(previous)
    monkeypatch.setattr(sys,'argv',['replay','--audit',str(tmp_path),'--compact']);replay_t030a.main()
    result=json.loads((tmp_path/'compact_replay.json').read_bytes())
    assert result['count']==1 and result['rows'][0]['selected_step']==14
    assert (tmp_path/'independent_replay.json').read_bytes()==previous
    assert not (d/'original.pt').exists() and not (d/'guarded.pt').exists()
