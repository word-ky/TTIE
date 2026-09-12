"""T017-B: attribute frozen T017-A reference-only failures; no new choices."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
from .local_geometry import CANDIDATES,choices,evaluate as evaluate_a,distribution,sha,write,now
from .routing.provenance import verify_source

CATEGORIES=('transfer_flip','soft_interaction_failure','zero/tie')


def quantities(candidate_mse,hard_index):
    soft=candidate_mse[1::3];s0=candidate_mse[13];s1=candidate_mse[hard_index+1]
    h0=candidate_mse[12];h1=candidate_mse[hard_index];minimum=min(soft)
    sg=s0-s1;hg=h0-h1;category=None
    if hg<0:category='transfer_flip' if sg>0 else 'soft_interaction_failure' if sg<0 else 'zero/tie'
    oracle=soft.index(minimum);chosen=hard_index//3
    return dict(S0=s0,S1=s1,H0=h0,H1=h1,S_star=minimum,soft_joint_gain=sg,hard_gain=hg,
        soft_separability_regret=s1-minimum,harmful_category=category,
        soft_oracle_index=oracle,soft_oracle_ties=soft.count(minimum),
        equals_first_soft_oracle=chosen==oracle,in_soft_oracle_tie_set=s1==minimum)


def dominance(counts,total):
    if total==0:return 'no_harmful_moves'
    if 3*counts['transfer_flip']>=2*total:return 'transfer_dominant'
    if 3*counts['soft_interaction_failure']>=2*total:return 'interaction_dominant'
    return 'mixed/inconclusive'


def interpretation(overall,quadrants):
    a=dominance(overall['category_counts'],overall['harmful_hard_moves'])
    q=dominance(quadrants['category_counts'],quadrants['harmful_hard_moves'])
    return a if a==q and a in ('transfer_dominant','interaction_dominant') else 'mixed/inconclusive'


MERGED_A='42bafee15964b75c9194e0b92d7f4bae23d049fc'
SOURCES=('ttie/__init__.py','ttie/routing/__init__.py','ttie/routing/provenance.py','ttie/local_geometry.py','ttie/geometry_attribution.py')


def load_inputs():
    data={};hashes={};raw_decisions=None
    for name in ('config','baseline','decisions','decisions_frozen','evaluation','summary','evaluation_receipt'):
        path='research_log/T017A_run/'+name+'.json';blob=subprocess.check_output(['git','show',MERGED_A+':'+path])
        data[name]=json.loads(blob);hashes[name]=dict(commit=MERGED_A,path=path,sha256=hashlib.sha256(blob).hexdigest())
        if name=='decisions':raw_decisions=blob
    for name in ('candidate_metrics','config'):
        item=data['config']['input_artifact_hashes'][name]
        assert item['commit']=='ee5d8fdaf3ab48ee7ad3654d45bdc65419be8367'
        blob=subprocess.check_output(['git','show',item['commit']+':'+item['path']])
        assert hashlib.sha256(blob).hexdigest()==item['sha256']
        data['table_'+name]=json.loads(blob);hashes['table_'+name]=item
    assert hashes['decisions']['sha256']==data['decisions_frozen']['decisions_sha256']==data['evaluation_receipt']['decisions_sha256']
    assert hashes['decisions_frozen']['sha256']==data['evaluation_receipt']['decisions_frozen_sha256']
    return data,hashes,raw_decisions


def reproduce_a(data):
    table=data['table_candidate_metrics'];assert len(table)==120 and all(len(r['candidate_mse'])==27 for r in table)
    assert data['table_config']['candidates']==[list(c) for c in CANDIDATES]
    assert choices(table)==data['decisions']
    report,rows=evaluate_a(data['decisions'],table)
    assert report==data['summary'] and rows==data['evaluation'] and report['passed']==4
    return dict(passed=True,completed_utc=now(),choices_exact=120,hard_mses_exact=120,summary_and_evaluation_exact=True,
        clauses=report['clauses'],decisions_sha256=data['decisions_frozen']['decisions_sha256'])


def measure(decisions,table):
    return [dict(row_index=d['row_index'],hard_index=d['hard_index'],**quantities(table[d['row_index']]['candidate_mse'],d['hard_index'])) for d in decisions]


def sign(value):return 'negative' if value<0 else 'positive' if value>0 else 'zero'


def report_groups(measured,table):
    rows=[dict(**r,condition=table[r['row_index']]['condition'],image_id=table[r['row_index']]['image_id']) for r in measured]
    groups={}
    for condition in ('spatial_pool','left_right','quadrants','offset_left_right_40'):
        group=rows if condition=='spatial_pool' else [r for r in rows if r['condition']==condition]
        total=sum(r['hard_gain']<0 for r in group);counts={c:sum(r['harmful_category']==c for r in group) for c in CATEGORIES}
        regret=distribution([r['soft_separability_regret'] for r in group])
        groups[condition]=dict(count=len(group),harmful_hard_moves=total,category_counts=counts,
            category_fractions={c:n/total if total else None for c,n in counts.items()},dominance=dominance(counts,total),
            sign_contingency={s:{h:sum(sign(r['soft_joint_gain'])==s and sign(r['hard_gain'])==h for r in group)
                                for h in ('negative','zero','positive')} for s in ('negative','zero','positive')},
            soft_separability_regret=dict(mean=regret['mean'],median=regret['median'],p95=regret['quantiles']['95'],
                                         max=regret['quantiles']['100'],zero_count=sum(r['soft_separability_regret']==0 for r in group)),
            first_soft_oracle_equal_count=sum(r['equals_first_soft_oracle'] for r in group),
            first_soft_oracle_equal_fraction=statistics.mean(r['equals_first_soft_oracle'] for r in group),
            in_soft_oracle_tie_set_count=sum(r['in_soft_oracle_tie_set'] for r in group),
            in_soft_oracle_tie_set_fraction=statistics.mean(r['in_soft_oracle_tie_set'] for r in group),
            soft_oracle_tied_episodes=sum(r['soft_oracle_ties']>1 for r in group),
            selected_in_tied_soft_oracle_count=sum(r['soft_oracle_ties']>1 and r['in_soft_oracle_tie_set'] for r in group))
        assert sum(counts.values())==total
    return dict(groups=groups,interpretation=interpretation(groups['spatial_pool'],groups['quadrants']),
        reference_only=True,qualification_experiment=False),rows


def finish(output,measured,table):
    write(output/'quantities.json',measured)
    write(output/'quantities_frozen.json',dict(finalized_utc=now(),quantities_sha256=sha(output/'quantities.json'),
        count=len(measured),family_labels_attached=False))
    started=now();report,rows=report_groups(measured,table)
    write(output/'summary.json',report);write(output/'attribution.json',rows)
    write(output/'report_receipt.json',dict(reporting_started_utc=started,quantities_sha256=sha(output/'quantities.json'),
        quantities_frozen_sha256=sha(output/'quantities_frozen.json'),original_decisions_sha256=sha(output/'frozen_A_decisions.json')))
    return report


def markdown(report):
    lines=['# T017-B: frozen-choice soft-to-hard failure attribution','',
        'Result: **'+report['interpretation']+'**. Reference-only development attribution, not a qualification or deployable selector.', '',
        '| Group | Harmful hard moves | Transfer / Interaction / Zero-tie | Fractions | Group dominance |',
        '|---|---:|---|---|---|']
    for c,g in report['groups'].items():lines.append(f"| {c} | {g['harmful_hard_moves']} | {g['category_counts']} | {g['category_fractions']} | {g['dominance']} |")
    lines+=['','Overall attribution requires the same category to reach at least 2/3 of harmful moves both pooled and in quadrants. Integer comparison 3*count>=2*total, no tolerance. No harmful moves gives undefined fractions, not dominance.', '',
        '| Group | Soft separability regret: mean / median / p95 / max / zero count | First oracle equality | In any oracle tie set | Tied oracle episodes / selected in tied set |',
        '|---|---|---|---|---|']
    for c,g in report['groups'].items():lines.append(f"| {c} | {g['soft_separability_regret']} | {g['first_soft_oracle_equal_count']} ({g['first_soft_oracle_equal_fraction']}) | {g['in_soft_oracle_tie_set_count']} ({g['in_soft_oracle_tie_set_fraction']}) | {g['soft_oracle_tied_episodes']} / {g['selected_in_tied_soft_oracle_count']} |")
    for c,g in report['groups'].items():
        lines+=['','## '+c+' gain signs','', 'Rows=soft_joint_gain sign; columns=hard_gain sign.', '', '| Soft sign | Hard negative | Hard zero | Hard positive |','|---|---:|---:|---:|']
        for s,v in g['sign_contingency'].items():lines.append('| '+s+' | '+' | '.join(str(v[h]) for h in ('negative','zero','positive'))+' |')
    meaning={'transfer_dominant':'The tau=.05 neighborhood is not a faithful surrogate for hard-boundary deployment; do not train a hard-deployment geometry objective from it in the next cycle.',
        'interaction_dominant':'Independent coordinates are inadequate inside the soft landscape; do not infer that a scalar per-axis derivative rule is sufficient.',
        'mixed/inconclusive':'Preserve ambiguity; do not select a preferred mechanism from secondary statistics.'}
    lines+=['','## Bounded interpretation','',meaning[report['interpretation']], '',
        'The nine-soft oracle is diagnostic only and never changes any of the 120 frozen A choices. Oracle equality includes exact tie-aware and first lexicographic variants separately. p95 uses unchanged linear quantiles. No clipping, tolerance, threshold, rescue rule or new selector. Stop after T017-B; no geometry objective or fresh training.','']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-sha',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    code=verify_source(a.source_sha,SOURCES);data,hashes,raw_decisions=load_inputs();baseline=reproduce_a(data)
    a.output.mkdir(parents=True);write(a.output/'baseline.json',baseline)
    (a.output/'frozen_A_decisions.json').write_bytes(raw_decisions)
    write(a.output/'config.json',dict(task='T017-B',source_sha=a.source_sha,source_code_sha256=code,input_artifact_hashes=hashes,
        reference_only=True,qualification_experiment=False,choice_mutations=0,dominance='same category >=2/3 overall and quadrants',
        runtime=dict(python=platform.python_version(),device='cpu'),merged_a_commit=MERGED_A))
    measured=measure(data['decisions'],data['table_candidate_metrics'])
    report=finish(a.output,measured,data['table_candidate_metrics'])
    assert sha(a.output/'frozen_A_decisions.json')==hashes['decisions']['sha256']
    (a.output/'T017B_analysis.md').write_text(markdown(report),encoding='utf-8')
    print(json.dumps(dict(interpretation=report['interpretation'],overall=report['groups']['spatial_pool']['category_counts'],quadrants=report['groups']['quadrants']['category_counts'])))


if __name__=='__main__':main()
