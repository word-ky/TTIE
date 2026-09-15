from research_log.T043A_quality.core import classify,aggregate

def test_joint_quality_gate():
    assert classify(.5,0)=='matched-gain early-state quality bridge supported'
    assert classify(.499,0)=='matched-gain early-state quality bridge not supported / mixed'
    assert classify(.6,-.00001)=='matched-gain early-state quality bridge not supported / mixed'

def test_paired_statistics_keep_losses():
    rows=[dict(index=i,low=str(i),baseline_psnr=10.,candidate_psnr=10.+d,baseline_ssim=.5,candidate_ssim=.5+d/100,delta_psnr=d,delta_ssim=d/100) for i,d in enumerate([-1.,0.,2.])]
    s=aggregate(rows);assert s['paired']['psnr']['positive']==s['paired']['psnr']['negative']==s['paired']['psnr']['zero']==1
    assert s['paired']['psnr']['worst'][0]['index']==0 and s['paired']['psnr']['best'][0]['index']==2
    assert s['classification']=='matched-gain early-state quality bridge not supported / mixed'
