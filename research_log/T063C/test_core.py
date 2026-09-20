import math
import numpy as np
from research_log.T063C.core import progress, choose, candidates, calibrate


def test_earliest_crossing_and_nonmonotone_minimum():
    values = [10., 8., 9., 6., 6., 7.]
    best, reduction, q = progress(values)
    assert best == 6 and reduction == 4
    np.testing.assert_array_equal(q, [0, .5, .25, 1, 1, .75])
    assert choose(values, 0) == 0
    assert choose(values, .5) == 1
    assert choose(values, .75) == 3
    assert choose(values, 1) == 3


def test_zero_progress_and_exact_grid():
    assert choose([1., 1., 1.], 1.) == 0
    assert choose([0., -1e-13], 1.) == 0
    assert candidates([[10., 12., 8., 6.], [4., 3., 0.]]) == [0., .25, .5, 1.]
    with np.errstate(invalid='ignore'):
        assert choose([math.inf, math.inf], .5) == 0


def test_threshold_arithmetic_not_rounded_progress_comparison():
    # The contract compares L_k to the reconstructed threshold, including binary64 rounding.
    values = [1., .9, .7, .5]
    rho = progress(values)[2][1]
    expected = next(k for k,v in enumerate(values) if v <= values[0] - rho*(values[0]-min(values)))
    assert choose(values, rho) == expected


def test_calibration_safety_and_smallest_rho_tie():
    rows = [dict(psnr=[0,14,14,1],ssim=[.5]*4,t026_psnr=10,t036_psnr=10,t036_ssim=.5)]*100
    table,best,passed=calibrate([[10.,8.,6.,0.]]*100,rows)
    assert passed and best['rho'] == .2
    assert not table[-1]['eligible']
    bad=[dict(psnr=[0]*4,ssim=[0]*4,t026_psnr=10,t036_psnr=10,t036_ssim=.5)]*100
    assert calibrate([[10.,8.,6.,0.]]*100,bad)[1:] == (None,False)
