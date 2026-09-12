"""T016-F: 25 fresh CPU rank30 heads, nested calibration, immutable evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import torch
from .boundary_rank_run import load_source,SOURCES as D_SOURCES,sha,write,now
from .boundary_rank import RECIPE
from .boundary_probe import probe_features,grouped_folds
from .boundary_probe_metrics import evaluate
from .boundary_confidence import THRESHOLDS,label
from .boundary_confidence_run import diagnostics
from .boundary_nested import run_outer
from .routing.provenance import verify_source

E='66b597cef47d483dea2027d9ad925d8961031f98'
SOURCES=D_SOURCES+('ttie/boundary_confidence.py','ttie/boundary_confidence_run.py','ttie/boundary_nested.py','ttie/boundary_nested_run.py')


def inputs():
    data,old,hashes=load_source();historical={}
    for key in ('config','summary','evaluation','decisions_frozen'):
        path='research_log/T016E_run/'+key+'.json';blob=subprocess.check_output(['git','show',E+':'+path])
        historical[key]=json.loads(blob);hashes['T016E/'+key]=dict(commit=E,path=path,sha256=hashlib.sha256(blob).hexdigest())
    for name,key in [('reference','T016B/reference'),('c_folds','T016C/folds'),('c_evaluation','T016C/evaluation')]:
        assert historical['config']['input_artifact_hashes'][name]==hashes[key]
    return data,old,historical,hashes


def evaluate_nested(decisions,reference,old_e):
    report,rows=evaluate(decisions,reference)
    ungated=[dict(row,selected_index=min(range(9),key=lambda i:row['predictions'][i])) for row in decisions]
    ungated_report,ungated_rows=evaluate(ungated,reference)
    old={r['episode']:r for r in old_e};base={r['episode']:r for r in ungated_rows};refs={r['episode']:r for r in reference}
    for row in rows:
        row['ungated_nested_mse']=base[row['episode']]['selected_mse'];row['old_t016e_mse']=old[row['episode']]['selected_mse']
        row['canonical_candidate_mse']=refs[row['episode']]['reference_mse'][4]
    for condition,g in report['groups'].items():
        group=rows if condition=='spatial_pool' else [r for r in rows if r['condition']==condition]
        for key in ('ungated_nested','old_t016e'):
            g[key+'_mse']=statistics.mean(r[key+'_mse'] for r in group)
            g['ratios']['selected_over_'+key]=g['selected_mse']/g[key+'_mse']
        g['confidence_diagnostics']=diagnostics(group)
    for fold in report['folds']:
        group=[r for r in rows if r['fold']==fold['fold']]
        fold['confidence_diagnostics']=diagnostics(group)
        fold['condition_diagnostics']={c:diagnostics([r for r in group if r['condition']==c]) for c in fold['per_condition']}
    report['canonical_vs_region2_max_abs']=max(abs(r['canonical_candidate_mse']-r['region2_mse']) for r in rows)
    report['interpretation']='nested_development_supports_rank30_single_confidence_fallback' if report['qualified'] else 't016e_pass_does_not_survive_fully_nested_development_audit'
    return report,rows,ungated_report,ungated_rows


def finish(output,decisions,fold_receipts,reference,old_e):
    write(output/'decisions.json',decisions)
    write(output/'all_folds_frozen.json',dict(finalized_utc=now(),decisions_sha256=sha(output/'decisions.json'),
        folds=fold_receipts,heldout_reference_evaluation=False))
    started=now()
    report,rows,ungated,ungated_rows=evaluate_nested(decisions,reference,old_e)
    write(output/'summary.json',report);write(output/'evaluation.json',rows)
    write(output/'ungated_summary.json',ungated);write(output/'ungated_evaluation.json',ungated_rows)
    write(output/'evaluation_receipt.json',dict(evaluation_started_utc=started,all_folds_frozen_sha256=sha(output/'all_folds_frozen.json'),
        decisions_sha256=sha(output/'decisions.json'),fold_freeze_hashes={str(k):sha(output/f'fold{k}'/'fold_frozen.json') for k in range(5)}))
    return report


def markdown(report,folds):
    lines=['# T016-F: fully nested rank30 + single confidence fallback','',
        f"Result: **{report['interpretation']}**, **{report['passed']}/5**.",'',
        'Five outer32/8ID splits; each outertrain32 split into four24/8ID innerfits. Exactly25freshheads using the unchangedDrecipe; no oldhead/score used for primarynested decisions.',
        'Thresholds: '+str([f['threshold'] for f in folds])+'.','',
        'Literalclauses: '+str(report['clauses']), '',
        '| Group | Selected MSE | /Region2 | /Hard oracle | /Ungated nested | /Old E | Adaptive / Canonical | Adapted gains |',
        '|---|---:|---:|---:|---:|---:|---|---|']
    for c,g in report['groups'].items():
        d=g['confidence_diagnostics'];values=[g['selected_mse']]+[g['ratios']['selected_over_'+k] for k in ('region2','hard_oracle','ungated_nested','old_t016e')]
        lines.append('| '+c+' | '+' | '.join(f'{x:.12g}' for x in values)+f" | {d['adaptive']} / {d['canonical']} | {d['adapted_gains']} |")
    lines+=['','| Group | q distribution | Disagreement / Outside oracle ties | Selected / Oracle counts |','|---|---|---|---|']
    for c,g in report['groups'].items():
        lines.append(f"| {c} | {g['confidence_diagnostics']['q']} | {g['disagreement_rate']} / {g['outside_oracle_tie_set_rate']} | {g['selected_counts']} / {g['oracle_counts']} |")
    lines+=['','| Fold | Threshold | Selected MSE | Adaptive / Canonical | Per-condition adaptive/canonical |','|---|---:|---:|---|---|']
    for f in report['folds']:
        d=f['confidence_diagnostics'];counts={c:(v['adaptive'],v['canonical']) for c,v in f['condition_diagnostics'].items()}
        lines.append(f"| {f['fold']} | {folds[f['fold']]['threshold']} | {f['selected_mse']:.12g} | {d['adaptive']} / {d['canonical']} | {counts} |")
    lines+=['','Per-head histories, pair counts, normalization, calibration tables, innerOOFs and allfold hashes are persisted. q uses the unchanged Epopulationstd/floor formula andlinear0/25/50/75/100percentiles for description only.',
        'Every fold excludes its outer-heldout IDs from allfiveheadtraining sets and thresholdcalibration. Each calibrationimage is also excluded from its own innerOOFranker training. All25heads andfivefolddecisions are frozen before combinedouterreferenceevaluation.',
        'Canonical candidate4 versus Region2 maxMSEdifference: '+str(report['canonical_vs_region2_max_abs'])+'.', '',
        'Development-only, on already-inspected40IDs. Historical E has inherited scorer dependence and is a comparator, not a qualified baseline. No fresh qualification or deployablethreshold claim. Stop after the literal result; no alternategrid,model,confidence orfollow-onwork.','']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-sha',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    code=verify_source(a.source_sha,SOURCES);torch.set_num_threads(1)
    data,old,historical,hashes=inputs();entries=data['selection']['episodes'];episodes=[r['episode'] for r in entries]
    reference={r['episode']:r for r in data['reference']};metadata=[dict(image_id=reference[e]['image_id']) for e in episodes]
    folds=grouped_folds(metadata);assert folds==old['folds']
    assert len(entries)==120 and all(len(f['train_image_ids'])==32 and len(f['heldout_image_ids'])==8 for f in folds)
    candidates=data['selection']['candidates'];assert candidates==[[x,y,0.] for x in (.4,.5,.6) for y in (.4,.5,.6)]
    x=probe_features([r['features'] for r in entries],candidates,30)
    a.output.mkdir(parents=True);write(a.output/'folds.json',folds)
    write(a.output/'config.json',dict(task='T016-F',source_sha=a.source_sha,source_code_sha256=code,input_artifact_hashes=hashes,
        recipe=dict(RECIPE,input_dim=30),thresholds=[label(t) for t in THRESHOLDS],inner_split='sorted outertrain numeric IDs, position mod4',
        outer_heads=5,inner_heads=20,primary_old_head_or_score_reuse=False,candidates=candidates,
        runtime=dict(python=platform.python_version(),torch=torch.__version__,device='cpu'),development_only=True))
    decisions=[];receipts=[]
    for fold in folds:
        rows,receipt=run_outer(x,metadata,episodes,reference,fold,a.output/f"fold{fold['fold']}")
        decisions.extend(rows);receipts.append(receipt)
        print('outer fold',fold['fold'],'five heads finalized; threshold',receipt['threshold'],flush=True)
    order={e:i for i,e in enumerate(episodes)};decisions.sort(key=lambda r:order[r['episode']])
    report=finish(a.output,decisions,receipts,data['reference'],historical['evaluation'])
    (a.output/'T016F_analysis.md').write_text(markdown(report,receipts),encoding='utf-8')
    print(json.dumps(dict(thresholds=[r['threshold'] for r in receipts],passed=report['passed'],interpretation=report['interpretation'])),flush=True)


if __name__=='__main__':main()
