import pytest,torch
from research_log.T058AE.storage import atomic_json,save_chunk,reopen,coverage
from research_log.T058A_tangent.core import thash

def test_offset_shard_roundtrip(tmp_path):
    gs=[torch.zeros(1,1,8,8),torch.ones(1,1,8,8)];chunk=save_chunk(tmp_path,[1024,1025],gs)
    rows=[dict(index=1024+i,gradient_sha256=thash(g),gradient_norm=float(g.double().norm())) for i,g in enumerate(gs)]
    atomic_json(tmp_path/'manifest.json',dict(rows=rows,chunks=[chunk]))
    assert reopen(tmp_path,2,1024)['last']==1025
    with pytest.raises(AssertionError):reopen(tmp_path,2,0)
    with pytest.raises(AssertionError):reopen(tmp_path,3,1024)
    assert not (tmp_path/'complete.json').exists()

def test_combined_gap_overlap_identity_order():
    banks=[dict(states=2,image_id=7),dict(states=2,image_id=9)]
    rows=[dict(index=i,bank_index=i//2,state_index=i%2,image_id=7 if i<2 else 9) for i in range(4)]
    assert coverage(rows[:2],rows[2:],banks)['unique_indices']==4
    for second in [rows[1:],rows[3:],list(reversed(rows[2:]))]:
        with pytest.raises(AssertionError):coverage(rows[:2],second,banks)
    bad=[dict(x) for x in rows[2:]];bad[0]['image_id']=8
    with pytest.raises(AssertionError):coverage(rows[:2],bad,banks)
