"""One target-free loss-balance rule; offline scalar calibration only."""
import math
import statistics
import numpy as np
from research_log.T062A.core import losses


def ratios(components):
    c = np.asarray(components, dtype=np.float64)
    p = 10 * c[:, 1] + 5 * c[:, 2]
    return (c[1:, 0] - c[0, 0]) / np.maximum(p[0] - p[1:], 1e-8)


def choose(r, tau):
    return max([0] + [k for k, v in enumerate(r, 1) if math.isfinite(v) and v <= tau])


def candidates(r):
    values = sorted({float(v) for row in r for v in row if math.isfinite(v)})
    return [math.nextafter(values[0], -math.inf)] + values + [math.nextafter(values[-1], math.inf)]


def summarize(rows, steps):
    delta = [r['psnr'][k] - r['t036_psnr'] for r, k in zip(rows, steps)]
    tail = [r['psnr'][k] - r['t026_psnr'] for r, k in zip(rows, steps)]
    ds = [r['ssim'][k] - r['t036_ssim'] for r, k in zip(rows, steps)]
    m = dict(mean_delta_psnr=statistics.fmean(delta), median_delta_psnr=statistics.median(delta),
             regressions_t026=sum(v < 0 for v in tail), worst_delta_t026=min(tail),
             mean_delta_ssim=statistics.fmean(ds))
    m['gates'] = dict(mean_psnr=m['mean_delta_psnr'] >= 2, median_psnr=m['median_delta_psnr'] > 0,
                      regressions=m['regressions_t026'] <= 29, worst=m['worst_delta_t026'] >= -5.614,
                      mean_ssim=m['mean_delta_ssim'] >= -.001)
    m['eligible'] = all(m['gates'][k] for k in ['regressions', 'worst', 'mean_ssim'])
    return m


def calibrate(r, rows):
    table = []
    for tau in candidates(r):
        steps = [choose(v, tau) for v in r]
        table.append(dict(tau=tau, **summarize(rows, steps)))
    eligible = [v for v in table if v['eligible']]
    best = max(eligible, key=lambda v: (v['mean_delta_psnr'], -v['tau'])) if eligible else None
    passed = best is not None and all(best['gates'].values())
    return table, best, passed
