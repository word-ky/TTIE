"""Exact normalized objective progress with earliest checkpoint crossing."""
import math
import numpy as np
from research_log.T062A.core import losses
from research_log.T063B.core import summarize


def progress(values):
    values = np.asarray(values, dtype=np.float64)
    best = float(values.min())
    reduction = float(values[0] - best)
    q = np.clip((values[0] - values) / max(reduction, 1e-12), 0, 1)
    return best, reduction, q


def choose(values, rho):
    _, reduction, _ = progress(values)
    if not math.isfinite(reduction) or reduction <= 1e-12:
        return 0
    threshold = float(values[0]) - rho * reduction
    for k, value in enumerate(values):
        if math.isfinite(value) and value <= threshold:
            return k
    return 0


def candidates(trajectories):
    return sorted({0., 1.} | {float(q) for values in trajectories for q in progress(values)[2] if math.isfinite(q)})


def calibrate(trajectories, rows):
    table = []
    for rho in candidates(trajectories):
        steps = [choose(values, rho) for values in trajectories]
        table.append(dict(rho=rho, **summarize(rows, steps)))
    eligible = [row for row in table if row['eligible']]
    best = max(eligible, key=lambda row: (row['mean_delta_psnr'], -row['rho'])) if eligible else None
    passed = best is not None and all(best['gates'].values())
    return table, best, passed
