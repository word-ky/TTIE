"""Reference-only metrics after the entire cohort's direction decisions freeze."""
import argparse
from pathlib import Path
import statistics
import torch
from ..natural import load_image
from ..soft_basis.renderer import FixedCorners
from .common import HARD, CONDITIONS, read, write, sha, now, tensor_sha, preflight, verify_prepared


def summarize(rows):
    groups={}
    for name in ('spatial_pool',*CONDITIONS):
        group=rows if name=='spatial_pool' else [r for r in rows if r['condition']==name]
        m={k:statistics.mean(r[k] for r in group) for k in ('H0','H1','H_star')}
        groups[name]=dict(count=len(group),mse=m,ratios=dict(H1_over_H0=m['H1']/m['H0'],H1_over_H_star=m['H1']/m['H_star']),
            outcomes=dict(beneficial=sum(r['H1']<r['H0'] for r in group),equal=sum(r['H1']==r['H0'] for r in group),harmful=sum(r['H1']>r['H0'] for r in group)),
            harmful_rows=[r['row_index'] for r in group if r['H1']>r['H0']])
        groups[name]['movements']={key:sum(r['movement']==key for r in group) for key in ('no_move','x_only','y_only','both')}
    p=groups['spatial_pool']['mse'];o=groups['offset_left_right_40']['mse'];l=groups['left_right']['mse'];q=groups['quadrants']['mse']
    clauses=dict(pooled_gain=p['H1']<=.97*p['H0'],pooled_oracle=p['H1']<=1.03*p['H_star'],offset_gain=o['H1']<=.95*o['H0'],
        left_right_safety=l['H1']<=1.01*l['H0'],quadrants_safety=q['H1']<=1.01*q['H0'])
    return dict(groups=groups,clauses=clauses,passed=sum(clauses.values()),fresh_qualified=all(clauses.values()),
        verdict='T019-D fresh qualification positive' if all(clauses.values()) else 'T019-D fresh qualification negative')


@torch.no_grad()
def evaluate(cohort,selected,images,output,source,device,replay_receipt):
    _,code=preflight(source);freeze=read(selected/'decisions_frozen.json')
    assert freeze['source_sha']==source and freeze['episodes']==120 and sha(selected/'decisions.json')==freeze['decisions_sha256']
    assert sha(selected/'config.json')==freeze['config_sha256'];decisions=read(selected/'decisions.json')
    replay=read(replay_receipt)
    assert replay['passed'] and replay['reference_reads'] is False and replay['exact_logits_classes_decisions']==120
    assert replay['decisions_sha256']==freeze['decisions_sha256'] and freeze['finalized_utc']<replay['completed_utc']
    opened=now();verify_prepared(cohort,read(selected/'config.json'));mapping=read(cohort/'mapping.json')
    assert sha(cohort/'manifest.json')==freeze['manifest_sha256'];manifest=read(cohort/'manifest.json')
    assert len(manifest['images'])==40 and len(mapping)==len(decisions)==120
    inputs=read(cohort/'inputs/index.json')['episodes'];rows=[];clean_cache={}
    output.mkdir(parents=True)
    for d,m,item in zip(decisions,mapping,inputs):
        assert d['row_index']==m['row_index']==item['row_index']
        path=images/m['filename'];assert sha(path)==m['source_sha256']
        if m['image_id'] not in clean_cache:clean_cache[m['image_id']]=load_image(path)
        clean=clean_cache[m['image_id']]
        assert sha(cohort/'inputs'/item['file'])==d['input_file_sha256']
        image=torch.load(cohort/'inputs'/item['file'],weights_only=True,map_location=device)['image']
        state=selected/f"{d['row_index']:03d}"/'state.pt';assert sha(state)==freeze['case_files_sha256'][str(d['row_index'])]['state.pt']
        corners=torch.load(state,weights_only=True,map_location=device)['corners'];assert tensor_sha(corners)==d['corners_sha256']
        mse=[float((FixedCorners(corners,c)(image).cpu()-clean).square().mean()) for c in HARD]
        assert HARD[d['score_index']]==(d['bx'],d['by'],0.) and d['hard_index']==3*d['score_index']
        h0=mse[4];h1=mse[d['score_index']];best=min(mse)
        rows.append(dict(**m,bx=d['bx'],by=d['by'],score_index=d['score_index'],candidate_mse=mse,H0=h0,H1=h1,H_star=best,
            movement='no_move' if d['bx']==.5 and d['by']==.5 else 'x_only' if d['by']==.5 else 'y_only' if d['bx']==.5 else 'both',
            delta=h1-h0,H1_over_H0=h1/h0,H1_over_H_star=h1/best))
    write(output/'evaluation.json',rows);report=summarize(rows);write(output/'summary.json',report)
    assert sha(selected/'decisions.json')==freeze['decisions_sha256']
    write(output/'evaluation_receipt.json',dict(reference_opened_utc=opened,completed_utc=now(),source_code_sha256=code,
        decision_freeze_sha256=sha(selected/'decisions_frozen.json'),decisions_sha256=freeze['decisions_sha256'],
        replay_receipt_sha256=sha(replay_receipt),
        evaluation_sha256=sha(output/'evaluation.json'),summary_sha256=sha(output/'summary.json'),episodes=120))
    print(report['verdict'],report['passed'],'/5',report['clauses'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('cohort','selected','images','output','replay-receipt'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--source-sha',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    evaluate(a.cohort,a.selected,a.images,a.output,a.source_sha,a.device,a.replay_receipt)
