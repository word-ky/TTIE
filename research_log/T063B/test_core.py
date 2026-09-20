import math
import numpy as np
from research_log.T063B.core import ratios, choose, candidates, calibrate


def test_ratio_denominator_and_largest_finite_choice():
    c = [[0, .2, .1], [.1, .1, .1], [.2, .3, .1], [.3, .2, .1]]
    np.testing.assert_allclose(ratios(c), [.1, 2e7, 3e7])
    assert choose([.2, math.nan, .1, math.inf, .3], .2) == 3
    assert choose([.2, .1], .09) == 0
    assert choose([.2, .1], .2) == 2


def test_exact_candidate_grid_and_sentinels():
    grid = candidates([[1, 2, 1, math.nan], [2, 3, math.inf]])
    assert grid == [math.nextafter(1., -math.inf), 1., 2., 3., math.nextafter(3., math.inf)]


def test_calibration_safety_and_smallest_tau_tie():
    rows = [dict(psnr=[0, 13, 14, 14, 1], ssim=[.5]*5, t026_psnr=10,
                 t036_psnr=10, t036_ssim=.5)]*100
    table, best, passed = calibrate([[1, 2, 3, 4]]*100, rows)
    assert passed and best['tau'] == 2 and best['mean_delta_psnr'] == 4
    assert not table[-1]['eligible']
    bad = [dict(psnr=[0]*5, ssim=[0]*5, t026_psnr=10, t036_psnr=10, t036_ssim=.5)]*100
    assert calibrate([[1,2,3,4]]*100, bad)[1:] == (None, False)
