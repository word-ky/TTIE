"""T007 fixed-score audit. Persist decisions before attaching offline truth."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

from .clip_signal import FrozenCLIP
from .clip_audit import audit_group
from .learned_prototypes import Prototypes, learned_scores
from .joint_gate import both_gates
from .natural import CONDITIONS, VIEW_NAMES, load_image, degrade
from .prototype_audit import localization


def inference(encoder, prototypes, image, receipt):
    scores, _ = learned_scores(encoder, prototypes, image)
    winner, baseline, joint, evidence = both_gates(scores, receipt)
    return [dict(d_dark=float(s[0]), d_bright=float(s[1]), type=('dark','bright')[w.item()],
                 active_baseline=b.item(), active_joint=j.item(), evidence=e.item())
            for s,w,b,j,e in zip(scores,winner,baseline,joint,evidence)]


def persist_then_label(rows, metadata, path):
    path.write_text(json.dumps(rows,indent=2))
    labeled=[]
    for row,meta in zip(rows,metadata):
        item=dict(row,**meta)
        kind=None
        if item['view']!='full' and item['condition'] in ('left_right','quadrants'):
            dark=('top_left','bottom_left') if item['condition']=='left_right' else ('top_left','bottom_right')
            kind='dark' if item['view'] in dark else 'bright'
        item['region_true_type']=kind
        item['region_margin']=None if kind is None else item['d_'+kind]-item['d_'+('bright' if kind=='dark' else 'dark')]
        labeled.append(item)
    return labeled


def summarize_joint(rows, score_identity=True):
    report={}
    for mode in ('baseline','joint'):
        mapped=[dict(r,active=r['active_'+mode]) for r in rows]
        groups=dict(all=mapped,full=[r for r in mapped if r['view']=='full'],
                    quadrants=[r for r in mapped if r['view']!='full'])
        homogeneous={}
        for name,group in groups.items():
            result=audit_group(group)
            clean=[r for r in group if r['condition']=='clean']
            ids={r['image_id'] for r in clean}
            result['clean_image_count']=len(ids)
            result['clean_image_any_activation']=sum(any(r['active'] for r in clean if r['image_id']==i) for i in ids)/len(ids)
            homogeneous[name]=result
        mixed=[r for r in mapped if r['region_true_type'] is not None]
        report[mode]=dict(homogeneous=homogeneous,localization=dict(all=localization(mixed),
            left_right=localization([r for r in mixed if r['condition']=='left_right']),
            quadrants=localization([r for r in mixed if r['condition']=='quadrants'])))
    counts={}
    for name,group in dict(all=rows,full=[r for r in rows if r['view']=='full'],
                          quadrants=[r for r in rows if r['view']!='full']).items():
        removed=lambda r:r['active_baseline'] and not r['active_joint']
        trade=dict(clean_activations_removed=sum(removed(r) for r in group if r['condition']=='clean'),
                   clean_activations_added=sum(r['active_joint'] and not r['active_baseline'] for r in group if r['condition']=='clean'))
        for kind in ('dark','bright'):
            trade['homogeneous_'+kind+'_correct_activations_lost']=sum(removed(r) and r['type']==kind for r in group if r['condition']=='homogeneous_'+kind)
            trade['mixed_'+kind+'_correct_activations_lost']=sum(removed(r) and r['type']==kind for r in group if r['region_true_type']==kind)
        counts[name]=trade
    h=report['joint']['homogeneous']['all'];m=report['joint']['localization']['all']
    precision=h['correct_type_among_all_active_homogeneous_degraded']
    criteria=dict(clean_fpr=h['clean_false_activation']<=.15,clean_image_any=h['clean_image_any_activation']<=.30,
                  dark_correct_tpr=h['dark']['tpr_correct_type']>=.50,bright_correct_tpr=h['bright']['tpr_correct_type']>=.50,
                  active_type_precision=precision is not None and precision>=.90,
                  mixed_dark_recall=m['dark']['correct_activation_recall']>=.50,mixed_bright_recall=m['bright']['correct_activation_recall']>=.50,
                  mixed_dark_wrong=m['dark']['wrong_type_activation_rate']<=.15,mixed_bright_wrong=m['bright']['wrong_type_activation_rate']<=.15,
                  frozen_score_type_identity=score_identity)
    auc_equal=all(report['baseline']['homogeneous'][g][k]['roc_auc']==report['joint']['homogeneous'][g][k]['roc_auc']
                  for g in ('all','full','quadrants') for k in ('dark','bright'))
    report.update(heldout_rows=len(rows),tradeoff_counts=counts,auc_identical=auc_equal,criteria=criteria,
                  qualifies_later_pilot=all(criteria.values()),failed_criteria=[k for k,v in criteria.items() if not v])
    return report


def summary_markdown(report):
    lines=['# T007 fixed joint calibration audit','',f"Qualifies later pilot: {report['qualifies_later_pilot']}; failed: {report['failed_criteria']}",'',
           '| Scope | Gate | Clean view FPR | Clean image-any | Dark AUC | Bright AUC | Dark correct TPR | Bright correct TPR | Active precision |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for scope in ('all','full','quadrants'):
        for mode in ('baseline','joint'):
            h=report[mode]['homogeneous'][scope]
            values=[h['clean_false_activation'],h['clean_image_any_activation'],h['dark']['roc_auc'],h['bright']['roc_auc'],h['dark']['tpr_correct_type'],h['bright']['tpr_correct_type'],h['correct_type_among_all_active_homogeneous_degraded']]
            lines.append('| '+scope+' | '+mode+' | '+' | '.join('null' if v is None else f'{v:.6f}' for v in values)+' |')
    lines+=['','| Mixed scope | Gate | True type | N | Correct recall | Wrong activation | Active precision |', '|---|---|---|---:|---:|---:|---:|']
    for scope in ('all','left_right','quadrants'):
        for mode in ('baseline','joint'):
            for kind in ('dark','bright'):
                m=report[mode]['localization'][scope][kind]
                lines.append(f"| {scope} | {mode} | {kind} | {m['view_count']} | {m['correct_activation_recall']:.6f} | {m['wrong_type_activation_rate']:.6f} | {m['correct_type_among_active']} |")
    lines+=['','## Activation tradeoff counts','','```json',json.dumps(report['tradeoff_counts'],indent=2),'```','',
            'All six AUC comparisons are identical: '+str(report['auc_identical'])+'. No retraining, threshold sweep or ISP adaptation.']
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser()
    for key in ('manifest','images','model-identity','prototypes','receipt','output','t006-images'):
        parser.add_argument('--'+key,required=True,type=Path)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    torch.manual_seed(7);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    args.output.mkdir(parents=True,exist_ok=True)
    receipt=json.loads(args.receipt.read_text());manifest=json.loads(args.manifest.read_text())
    identity=json.loads(args.model_identity.read_text())
    digest=hashlib.sha256(args.prototypes.read_bytes()).hexdigest()
    assert digest==receipt['prototype_identity']['sha256']
    assert identity['sha256']==receipt['model_identity']['sha256']
    encoder=FrozenCLIP.from_checkpoint(identity['path'],args.device)
    saved=torch.load(args.prototypes,map_location=args.device,weights_only=True)
    prototypes=Prototypes(saved['raw']).eval().requires_grad_(False)
    old=Path('research_log/remote_runs/20260912-071126-ttie-t006-a6000/artifacts/audit')
    calibration_rows=json.loads((old/'calibration_scores.json').read_text())
    first=receipt['image_ids'][0]
    check_image=load_image(args.t006_images/f'{first:012d}.jpg').to(args.device)
    actual=learned_scores(encoder,prototypes,check_image)[0].cpu()
    expected=torch.tensor([[r['d_dark'],r['d_bright']] for r in calibration_rows if r['image_id']==first])
    max_delta=(actual-expected).abs().max().item()
    checks=dict(prototype_sha256=digest,calibration_score_repeat_max_delta=max_delta,
                calibration_score_bitwise_equal=torch.equal(actual,expected),
                frozen=all(not p.requires_grad and p.grad is None for module in (encoder,prototypes) for p in module.parameters()))
    assert max_delta<=1e-6 and checks['frozen']
    (args.output/'pre_evaluation_identity.json').write_text(json.dumps(checks,indent=2))
    (args.output/'config.json').write_text(json.dumps(dict(manifest=manifest,receipt=receipt,model_identity=identity,
               geometry=encoder.preprocess,conditions=CONDITIONS,device=args.device,dtype='float32',seed=7),indent=2))
    rows=[];metadata=[]
    for entry in manifest['images']:
        clean=load_image(args.images/entry['filename']).to(args.device)
        for condition in CONDITIONS:
            current=inference(encoder,prototypes,degrade(clean,condition),receipt)
            rows.extend(current)
            metadata.extend(dict(image_id=entry['image_id'],split=entry['split'],condition=condition,view=v) for v in VIEW_NAMES)
        print('Scored held-out',entry['image_id'],flush=True)
    labeled=persist_then_label(rows,metadata,args.output/'scores_before_metadata.json')
    assert all(all(item[k]==r[k] for k in r) for item,r in zip(labeled,rows))
    assert all(torch.isfinite(torch.tensor([r['d_dark'],r['d_bright'],r['evidence']])).all() for r in rows)
    (args.output/'scores.json').write_text(json.dumps(labeled,indent=2))
    with (args.output/'scores.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(labeled[0]));writer.writeheader();writer.writerows(labeled)
    report=summarize_joint(labeled,score_identity=max_delta<=1e-6)
    (args.output/'summary.json').write_text(json.dumps(report,indent=2))
    (args.output/'summary.md').write_text(summary_markdown(report))
    print(json.dumps(report,indent=2),flush=True)
    print('T007 completed; stop regardless of gate. No ISP adaptation.',flush=True)


if __name__=='__main__':main()
