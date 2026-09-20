import hashlib,pytest
from research_log.T062CR1.prepare import select

def test_union_exclusion_and_exact_hash_order():
    rows=[dict(low=f'Train/Low/low{i:05d}.png',normal=f'Train/Normal/normal{i:05d}.png') for i in range(200)]
    used={r['normal'] for r in rows[:75]}
    expected=sorted(rows[75:],key=lambda r:hashlib.sha256(('T062C-R1:'+r['low']).encode()).digest())[:100]
    actual=select(rows,used)
    assert [r['low'] for r in actual]==[r['low'] for r in expected]
    assert not used&{r['normal'] for r in actual}

def test_insufficient_untouched_blocks():
    with pytest.raises(AssertionError):select([dict(low='Train/Low/low1.png',normal='Train/Normal/normal1.png')],set())

def test_ignores_quality_and_conditions():
    rows=[dict(low=f'Train/Low/low{i:05d}.png',normal=f'Train/Normal/normal{i:05d}.png') for i in range(120)]
    assert select(rows,set())==select([dict(**r,psnr=-999,condition='ignored') for r in rows],set())
