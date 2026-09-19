import hashlib
from research_log.T062C.prepare import choose

def test_literal_hash_order_and_exclusion():
    records=[dict(low=f'Train/Low/low{i:05d}.png',normal=f'Train/Normal/normal{i:05d}.png') for i in range(160)]
    excluded={'low00001.png','low00042.png'}
    got=choose(records,excluded)
    expected=sorted((r for r in records if r['low'].split('/')[-1] not in excluded),key=lambda r:hashlib.sha256(('T062C-v1:'+r['low']).encode()).digest())[:100]
    assert [r['low'] for r in got]==[r['low'] for r in expected]
    assert len(got)==100

def test_prior_use_does_not_change_selection():
    records=[dict(low=f'Train/Low/low{i:05d}.png',normal=f'Train/Normal/normal{i:05d}.png',excluded_by=['prior use']) for i in range(120)]
    assert choose(records,set())==choose([{k:v for k,v in r.items() if k!='excluded_by'} for r in records],set())
