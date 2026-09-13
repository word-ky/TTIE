"""Reference-only metrics after the entire cohort's direction decisions freeze."""
import argparse
from pathlib import Path
import statistics
import torch
from ..natural import load_image
from ..soft_basis.renderer import FixedCorners
from .common import HARD, CONDITIONS, read, write, sha, now, tensor_sha, preflight, verify_prepared


def ratio(h1, h0):
    # Identity can have exactly zero clean MSE; report an undefined ratio as null.
    # The acceptance clauses always compare the absolute means directly.
    return h1/h0 if h0 != 0 else None


def p95(values):
    ordered=sorted(values);position=.95*(len(ordered)-1);lower=int(position)
    return ordered[lower]+(position-lower)*(ordered[min(lower+1,len(ordered)-1)]-ordered[lower])


def summarize(rows):
    groups={}
    for name in ('nonspatial_pool',*CONDITIONS):
        group=rows if name=='nonspatial_pool' else [r for r in rows if r['condition']==name]
        m={k:statistics.mean(r[k] for r in group) for k in ('H0','H1')}
        groups[name]=dict(count=len(group),mse=m,ratios=dict(H1_over_H0=ratio(m['H1'],m['H0'])),
            outcomes=dict(beneficial=sum(r['H1']<r['H0'] for r in group),equal=sum(r['H1']==r['H0'] for r in group),harmful=sum(r['H1']>r['H0'] for r in group)),
            harmful_rows=[r['row_index'] for r in group if r['H1']>r['H0']])
        groups[name]['movements']={key:sum(r['movement']==key for r in group) for key in ('no_move','x_only','y_only','both')}
    clauses={name+'_safety':group['mse']['H1']<=1.01*group['mse']['H0'] for name,group in groups.items()}
    clean=[r for r in rows if r['condition']=='clean']
    groups['clean']['absolute_mse']={k:dict(mean=groups['clean']['mse'][k],p95=p95([r[k] for r in clean])) for k in ('H0','H1')}
    return dict(groups=groups,clauses=clauses,passed=sum(clauses.values()),fresh_qualified=all(clauses.values()),
        p95_convention='linear interpolation at 0.95*(n-1) in sorted per-episode MSE',
        verdict='T020-A fresh safety positive' if all(clauses.values()) else 'T020-A fresh safety negative')


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
        assert HARD[d['score_index']]==(d['bx'],d['by'],0.) and d['hard_index']==3*d['score_index']
        h0=float((FixedCorners(corners,(.5,.5,0.))(image).cpu()-clean).square().mean())
        h1=float((FixedCorners(corners,(d['bx'],d['by'],0.))(image).cpu()-clean).square().mean())
        rows.append(dict(**m,bx=d['bx'],by=d['by'],score_index=d['score_index'],H0=h0,H1=h1,
            movement='no_move' if d['bx']==.5 and d['by']==.5 else 'x_only' if d['by']==.5 else 'y_only' if d['bx']==.5 else 'both',
            delta=h1-h0,H1_over_H0=ratio(h1,h0)))
    write(output/'evaluation.json',rows);report=summarize(rows);write(output/'summary.json',report)
    assert sha(selected/'decisions.json')==freeze['decisions_sha256']
    write(output/'evaluation_receipt.json',dict(reference_opened_utc=opened,completed_utc=now(),source_code_sha256=code,
        decision_freeze_sha256=sha(selected/'decisions_frozen.json'),decisions_sha256=freeze['decisions_sha256'],
        replay_receipt_sha256=sha(replay_receipt),
        evaluation_sha256=sha(output/'evaluation.json'),summary_sha256=sha(output/'summary.json'),episodes=120))
    print(report['verdict'],report['passed'],'/4',report['clauses'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('cohort','selected','images','output','replay-receipt'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--source-sha',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    evaluate(a.cohort,a.selected,a.images,a.output,a.source_sha,a.device,a.replay_receipt)
