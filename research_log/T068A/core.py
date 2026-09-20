CAPS=list(range(28))
LAMBDA=.875

def capped_step(first_safe,interior,cap):
    return max(first_safe,min(interior,cap))

def select(table):
    passing=[r for r in table if all(r['gates'].values())]
    order=sorted(passing,key=lambda r:(r['worst_delta_t026'],r['mean_delta_psnr'],r['median_delta_psnr'],r['K']),reverse=True)
    assert order,'No passing K: accepted no-cap control must pass'
    best=order[0]
    verdict='ABS_STEP_CAP_DEV_CANDIDATE_FROZEN' if best['K']<27 else 'ABS_STEP_CAP_DEV_NO_GAIN'
    return best,verdict,[r['K'] for r in order]
