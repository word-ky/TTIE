from research_log.T062B.analyze import select,summary

def test_no_safe_step():
    assert select([dict(step=k,safety_eligible=False) for k in range(41)])['k_star'] is None
    assert select([dict(step=0,safety_eligible=False)])['verdict']=='NEGATIVE'

def test_eligibility_before_max_and_earliest_tie():
    table=[dict(step=0,safety_eligible=False,mean_delta=20,median_delta=10),dict(step=1,safety_eligible=True,mean_delta=3,median_delta=1),dict(step=2,safety_eligible=True,mean_delta=3,median_delta=1)]
    r=select(table);assert r['k_star']==1 and r['verdict']=='PASS'
    table[1]['mean_delta']=1;table[2]['mean_delta']=1
    assert select(table)['verdict']=='NEGATIVE'

def test_safety_boundary():
    rows=[dict(psnr=[12.],ssim=[.5],t036_psnr=10.,t036_ssim=.5,t026_psnr=10.) for _ in range(100)]
    r=summary(rows,0);assert r['safety_eligible'] and r['mean_delta']==2 and r['improve']==100
    for row in rows:row['ssim']=[.49]
    assert not summary(rows,0)['safety_eligible']
