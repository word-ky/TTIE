import numpy as np
from research_log.T071B.common import summary
from research_log.T071A.core import metrics
from scripts.evaluate_t026a import independent_ssim

def test_paired_mean_and_median_are_not_difference_of_medians():
    rows=[dict(psnr=b,rgb_ssim=.5,ours_minus_baseline_psnr=a-b) for a,b in zip([1,2,100],[0,90,91])]
    got=summary(rows)
    assert got['ours_minus_baseline_median_psnr']==1
    assert got['ours_minus_baseline_mean_psnr']==np.mean([1,-88,9])

def test_exact_t071_metric_matches_independent_formula():
    x=np.random.default_rng(7).random((21,24,3)).astype(np.float32).astype(np.float64)
    y=x*.8
    m=metrics(x,y)
    assert abs(m['psnr']-(-10*np.log10(np.mean((x-y)**2))))<1e-12
    assert abs(m['rgb_ssim']-independent_ssim(x,y))<1e-12


def test_official_manifest_provenance_and_order():
    from research_log.T071B.common import read,PAIRS
    c=read(PAIRS)
    assert c['provenance']['archive_sha256']=='9820d8b112438d94d1f5d4d25817eee618a8cf4bc63a65cd6b19f5a98c3faefa'
    assert [r['image_number'] for r in c['rows']]==list(range(690,790))
