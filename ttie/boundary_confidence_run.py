"""T016-E fixed CPU confidence audit on immutable T016-D OOF scores."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
from .boundary_confidence import THRESHOLDS,label,calibrate_fold
from .boundary_probe_metrics import evaluate
from .routing.provenance import verify_source

D='aa71d268294e35f5df67c76eada29f9bec117abe'
C='433683eccad24dc763450be6a546072a72e0910b'
B='4062e01cb93de731c394015c5ac741d6c08e04d8'
INPUTS={
    'scores':(D,'research_log/T016D_run/rank30/oof.json'),
    'folds':(D,'research_log/T016D_run/folds.json'),
    'd_config':(D,'research_log/T016D_run/config.json'),
    'd_frozen':(D,'research_log/T016D_run/OOF_frozen.json'),
    'd_evaluation':(D,'research_log/T016D_run/evaluation.json'),
    'c_folds':(C,'research_log/T016C_run/folds.json'),
    'c_evaluation':(C,'research_log/T016C_run/evaluation.json'),
    'reference':(B,'research_log/remote_runs/20260913-013748-ttie-t016b-assets-ready/artifacts/evaluation/evaluation.json')}
SOURCES=('ttie/__init__.py','ttie/routing/__init__.py','ttie/routing/provenance.py',
    'ttie/boundary_probe_metrics.py','ttie/boundary_confidence.py','ttie/boundary_confidence_run.py')


def now():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def load_inputs():
    data={}; hashes={}
    for name,(commit,path) in INPUTS.items():
        blob=subprocess.check_output(['git','show',commit+':'+path]);data[name]=json.loads(blob)
        hashes[name]=dict(commit=commit,path=path,sha256=hashlib.sha256(blob).hexdigest())
    assert hashes['scores']['sha256']==data['d_frozen']['probes']['rank30']['oof_sha256']
    for name,key in [('reference','T016B/reference'),('c_folds','T016C/folds'),('c_evaluation','T016C/evaluation')]:
        assert hashes[name]==data['d_config']['input_artifact_hashes'][key]
    assert data['folds']==data['c_folds']
    assert data['d_config']['candidates']==[[x,y,0.] for x in (.4,.5,.6) for y in (.4,.5,.6)]
    rows=data['scores'];assert len(rows)==120 and len({r['episode'] for r in rows})==120
    for fold in data['folds']:
        assert len(fold['train_image_ids'])==32 and len(fold['heldout_image_ids'])==8
        assert not set(fold['train_image_ids'])&set(fold['heldout_image_ids'])
        assert all(rows[i]['fold']==fold['fold'] for i in fold['heldout'])
        assert all(rows[i]['fold']!=fold['fold'] for i in fold['train'])
    return data,hashes


def q_summary(values):
    ordered=sorted(values)
    def quantile(p):
        index=(len(ordered)-1)*p; lo=int(index); hi=min(lo+1,len(ordered)-1)
        return ordered[lo]+(ordered[hi]-ordered[lo])*(index-lo)
    return dict(count=len(values),mean=statistics.mean(values),median=statistics.median(values),
        quantiles={str(int(p*100)):quantile(p) for p in (0.,.25,.5,.75,1.)})


def diagnostics(rows):
    adapted=[r for r in rows if r['selected_index']!=4]
    return dict(count=len(rows),adaptive=len(adapted),canonical=len(rows)-len(adapted),q=q_summary([r['q'] for r in rows]),
        adapted_gains=dict(beneficial=sum(r['selected_mse']<r['canonical_candidate_mse'] for r in adapted),
                          harmful=sum(r['selected_mse']>r['canonical_candidate_mse'] for r in adapted),
                          zero=sum(r['selected_mse']==r['canonical_candidate_mse'] for r in adapted)))


def evaluate_decisions(decisions,reference,ungated,pointwise):
    report,rows=evaluate(decisions,reference)
    refs={r['episode']:r for r in reference}; old={r['episode']:r for r in ungated}; c={r['episode']:r for r in pointwise}
    for row in rows:
        row['ungated_rank30_mse']=old[row['episode']]['selected_mse']
        row['pointwise_probe30_mse']=c[row['episode']]['selected_mse']
        row['canonical_candidate_mse']=refs[row['episode']]['reference_mse'][4]
    report['canonical_candidate_vs_region2_max_abs']=max(abs(r['canonical_candidate_mse']-r['region2_mse']) for r in rows)
    for condition,g in report['groups'].items():
        group=rows if condition=='spatial_pool' else [r for r in rows if r['condition']==condition]
        for key in ('ungated_rank30','pointwise_probe30'):
            g[key+'_mse']=statistics.mean(r[key+'_mse'] for r in group)
            g['ratios']['selected_over_'+key]=g['selected_mse']/g[key+'_mse']
        g['confidence_diagnostics']=diagnostics(group)
    for fold in report['folds']:
        group=[r for r in rows if r['fold']==fold['fold']]
        fold['confidence_diagnostics']=diagnostics(group)
        fold['condition_diagnostics']={c:diagnostics([r for r in group if r['condition']==c]) for c in fold['per_condition']}
    report['interpretation']='rank30_development_safe_with_training_only_confidence_fallback' if report['qualified'] else 'simple_scalar_confidence_abstention_does_not_safely_unlock_boundary_headroom'
    return report,rows


def finish(output,calibrations,decisions,reference,ungated,pointwise):
    write(output/'calibrations.json',calibrations);write(output/'decisions.json',decisions)
    write(output/'decisions_frozen.json',dict(finalized_utc=now(),thresholds=[r['threshold'] for r in calibrations],
        decisions_sha256=sha(output/'decisions.json'),calibrations_sha256=sha(output/'calibrations.json'),
        heldout_reference_evaluation=False))
    started=now();report,rows=evaluate_decisions(decisions,reference,ungated,pointwise)
    write(output/'summary.json',report);write(output/'evaluation.json',rows)
    write(output/'evaluation_receipt.json',dict(evaluation_started_utc=started,frozen_sha256=sha(output/'decisions_frozen.json'),
        decisions_sha256=sha(output/'decisions.json'),calibrations_sha256=sha(output/'calibrations.json')))
    return report


def markdown(report,calibrations):
    lines=['# T016-E: fixed OOF confidence-abstention audit','',
        f"Result: **{report['interpretation']}**; **{report['passed']}/5** clauses.",'',
        'Thresholds by outer fold: '+str([r['threshold'] for r in calibrations])+'. Infinity always selects canonical.',
        '', 'Literal clauses: '+str(report['clauses']), '',
        '| Group | MSE | /Region2 | /Hard oracle | /Ungated rank30 | /Pointwise probe30 | Adaptive / Canonical | Beneficial / Harmful / Zero among adapted |',
        '|---|---:|---:|---:|---:|---:|---|---|']
    for c,g in report['groups'].items():
        d=g['confidence_diagnostics'];v=[g['selected_mse']]+[g['ratios']['selected_over_'+k] for k in ('region2','hard_oracle','ungated_rank30','pointwise_probe30')]
        lines.append('| '+c+' | '+' | '.join(f'{x:.12g}' for x in v)+f" | {d['adaptive']} / {d['canonical']} | {d['adapted_gains']} |")
    lines+=['','| Group | q summary (linear quantiles0/25/50/75/100) | Disagreement | Outside oracle ties | Selected / Oracle counts |','|---|---|---:|---:|---|']
    for c,g in report['groups'].items():
        lines.append(f"| {c} | {g['confidence_diagnostics']['q']} | {g['disagreement_rate']} | {g['outside_oracle_tie_set_rate']} | {g['selected_counts']} / {g['oracle_counts']} |")
    lines+=['','| Fold | Threshold | Selected MSE | Adaptive / Canonical | Per-condition adaptive / canonical |','|---|---:|---:|---|---|']
    for f in report['folds']:
        d=f['confidence_diagnostics'];counts={c:(g['adaptive'],g['canonical']) for c,g in f['condition_diagnostics'].items()}
        lines.append(f"| {f['fold']} | {calibrations[f['fold']]['threshold']} | {f['selected_mse']:.12g} | {d['adaptive']} / {d['canonical']} | {counts} |")
    lines+=['','All eight calibration means per fold, exact q/decision values and per-fold/condition diagnostics are in the accompanying JSON artifacts.',
        'Canonical candidate4 versus accepted Region2 maximum MSE difference: '+str(report['canonical_candidate_vs_region2_max_abs'])+'.','',
        'Only the fixed q and eight thresholds were used. Strict q>t adapts; equal q falls back; equal calibration means choose the larger threshold. Calibration uses the other32IDs and their own pre-existing OOF scores; no model loaded or retrained. All five thresholds and120decisions were frozen before evaluation.', '',
        'Protocol limitation: this is the requested reuse of OOF scores, not fully nested ranker training. A calibration image\'s frozen D ranker may have trained on IDs held out by the current outer threshold fold. Direct threshold fitting never reads those held-out reference rows, but inherited scorer-training dependence remains. Development-only; no independent fresh-generalization claim.', '',
        'Stop after the literal result. No alternate threshold/grid, second gate, fresh data or follow-on experiment.','']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-sha',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    source=verify_source(a.source_sha,SOURCES);data,hashes=load_inputs()
    a.output.mkdir(parents=True)
    write(a.output/'config.json',dict(task='T016-E',source_sha=a.source_sha,source_code_sha256=source,input_artifact_hashes=hashes,
        thresholds=[label(t) for t in THRESHOLDS],canonical_index=4,q='(s[4]-min_noncanonical(s))/max(population_std(s),1e-12); constant=0',
        comparator='strict greater',calibration_tie='larger threshold',calibration_objective='unweighted mean MSE over96episodes',
        runtime=dict(python=platform.python_version(),device='cpu',torch_imported=False),retraining=False,development_only=True,
        q_quantiles=[0,25,50,75,100],quantile_interpolation='linear'))
    write(a.output/'folds.json',data['folds'])
    reference={r['episode']:r for r in data['reference']};calibrations=[];decisions=[]
    for fold in data['folds']:
        calibration,heldout=calibrate_fold(data['scores'],fold,reference)
        calibration['input_oof_sha256']=hashes['scores']['sha256'];calibration['finalized_utc']=now()
        calibrations.append(calibration);decisions.extend(heldout)
    order={r['episode']:i for i,r in enumerate(data['scores'])};decisions.sort(key=lambda r:order[r['episode']])
    report=finish(a.output,calibrations,decisions,data['reference'],data['d_evaluation']['rank30'],data['c_evaluation']['probe30'])
    (a.output/'T016E_analysis.md').write_text(markdown(report,calibrations),encoding='utf-8')
    print(json.dumps(dict(thresholds=[r['threshold'] for r in calibrations],passed=report['passed'],interpretation=report['interpretation'])))


if __name__=='__main__':main()
