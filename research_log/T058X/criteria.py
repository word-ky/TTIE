from research_log.T058V.numerics import ulp

def compare(a,b,kind):
    tolerance=max(1e-6,4*max(ulp(a),ulp(b))) if kind=='primal' else 1e-6+1e-4*abs(a)
    error=abs(a-b)
    return dict(error=error,tolerance=tolerance,margin=tolerance-error,passed=error<=tolerance)
