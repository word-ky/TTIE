import numpy as np
from research_log.T069B.core import TOTAL_ATOL,TOTAL_RTOL,score

def comparison(actual,anchor):
    a=np.asarray(actual,dtype=np.float64);b=np.asarray(anchor,dtype=np.float64)
    assert a.shape==b.shape==(12,) and np.isfinite(a).all() and np.isfinite(b).all()
    residual=np.abs(a-b);failed=residual>TOTAL_ATOL+TOTAL_RTOL*np.abs(b)
    return dict(abs_residual=residual.tolist(),max_abs=float(residual.max()),l2_relative=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-12)),failed_coordinates=int(failed.sum()),passes=not bool(failed.any()))

def bookkeeping(components,direct,stored=None):
    stats=score(components);direct=np.asarray(direct,dtype=np.float64)
    assert np.isfinite(np.asarray(components)).all() and np.isfinite(direct).all()
    den=stats['sum_component_norms'];rd=1-float(np.linalg.norm(direct))/max(den,1e-12)
    result=dict(sum_vs_direct=comparison(stats['summed_gradient'],direct),R_cancel=stats['R_cancel'],R_directnorm=rd,score_sensitivity=abs(stats['R_cancel']-rd))
    if stored is not None:result['direct_vs_trace']=comparison(direct,stored)
    return result

def audit_status(rows):
    return 'GRADIENT_NUMERICS_CHARACTERIZED' if all(r['float32']['direct_vs_trace']['passes'] and r['repeat']['direct_comparison']['passes'] for r in rows) else 'GRADIENT_PATH_MISMATCH'

def distribution(values):
    a=np.asarray(values,dtype=np.float64)
    return dict(max=float(a.max()),median=float(np.median(a)),p95=float(np.percentile(a,95,method='linear')))

def summarize(rows):
    result=dict(status=audit_status(rows),rows=len(rows),reference_reads=0,optimizer_runs=0,model_fits=0)
    for name in ['direct_vs_trace','sum_vs_direct']:
        cs=[r['float32'][name] for r in rows]
        result[name]=dict(absolute_coordinates=distribution([x for c in cs for x in c['abs_residual']]),absolute_row_max=distribution([c['max_abs'] for c in cs]),l2_relative=distribution([c['l2_relative'] for c in cs]),failed_rows=sum(not c['passes'] for c in cs),failed_coordinates=sum(c['failed_coordinates'] for c in cs))
    result['score_sensitivity']=distribution([r['float32']['score_sensitivity'] for r in rows])
    result['repeat']=dict(exact_direct_rows=sum(r['repeat']['direct_equal'] for r in rows),exact_component_rows=sum(all(r['repeat']['components_equal']) for r in rows),direct_tolerance_failed_rows=sum(not r['repeat']['direct_comparison']['passes'] for r in rows))
    ds=[r['float64']['audit'] for r in rows if r['float64']['status']=='supported']
    result['float64']=dict(supported_rows=len(ds),unsupported_rows=len(rows)-len(ds))
    if ds:result['float64'].update(sum_vs_direct_abs=distribution([x for d in ds for x in d['sum_vs_direct']['abs_residual']]),sum_vs_direct_l2_relative=distribution([d['sum_vs_direct']['l2_relative'] for d in ds]),score_sensitivity=distribution([d['score_sensitivity'] for d in ds]))
    return result
