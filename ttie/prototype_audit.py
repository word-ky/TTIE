"""T006 fixed source training and fresh homogeneous/mixed signal audit."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

from .clip_signal import FrozenCLIP, PROMPTS, decisions
from .clip_audit import audit_group, score_rows
from .natural import CONDITIONS, VIEW_NAMES, load_image, degrade
from .learned_prototypes import TRAIN_CONDITIONS, TRAIN_CONFIG, Prototypes, train_prototypes, source_calibration, learned_scores


def apply_decisions(rows, calibration, zero_calibration):
    for row in rows:
        winner, active=decisions(torch.tensor([[row['d_dark'],row['d_bright']]]),calibration)
        zw, za=decisions(torch.tensor([[row['zero_d_dark'],row['zero_d_bright']]]),zero_calibration)
        row.update(type=('dark','bright')[winner.item()],active=active.item(),
                   zero_type=('dark','bright')[zw.item()],zero_active=za.item())


def attach_regions(rows):
    """Offline ONLY: called after all pixel scores/decisions have been saved."""
    for row in rows:
        kind=None
        if row['view']!='full' and row['condition'] in ('left_right','quadrants'):
            dark_views=('top_left','bottom_left') if row['condition']=='left_right' else ('top_left','bottom_right')
            kind='dark' if row['view'] in dark_views else 'bright'
        row['region_true_type']=kind
        if kind is not None:
            other='bright' if kind=='dark' else 'dark'
            row['region_margin']=row['d_'+kind]-row['d_'+other]
            row['zero_region_margin']=row['zero_d_'+kind]-row['zero_d_'+other]
        else:
            row['region_margin']=row['zero_region_margin']=None


def localization(rows, *, zero=False):
    result={}
    for kind in ('dark','bright'):
        group=[r for r in rows if r['region_true_type']==kind]
        active=[r for r in group if r['zero_active' if zero else 'active']]
        correct=sum(r['zero_type' if zero else 'type']==kind for r in active)
        margins=torch.tensor([r['zero_region_margin' if zero else 'region_margin'] for r in group],dtype=torch.float64)
        quantiles=torch.quantile(margins,torch.tensor([0,.25,.5,.75,1.],dtype=torch.float64)).tolist()
        result[kind]=dict(view_count=len(group),correct_activation_recall=correct/len(group),
                          wrong_type_activation_rate=(len(active)-correct)/len(group),
                          correct_type_among_active=correct/len(active) if active else None,
                          margin_mean=margins.mean().item(),margin_min=quantiles[0],margin_p25=quantiles[1],
                          margin_median=quantiles[2],margin_p75=quantiles[3],margin_max=quantiles[4])
    return result


def summarize_prototypes(rows):
    evaluation=[r for r in rows if r['split']=='evaluation']
    def homogeneous(items):
        return dict(all=audit_group(items),full=audit_group([r for r in items if r['view']=='full']),
                    quadrants=audit_group([r for r in items if r['view']!='full']))
    zero=[dict(r,d_dark=r['zero_d_dark'],d_bright=r['zero_d_bright'],active=r['zero_active'],type=r['zero_type']) for r in evaluation]
    mixed=[r for r in evaluation if r['region_true_type'] is not None]
    report=dict(heldout_rows=len(evaluation),learned=homogeneous(evaluation),zero_shot=homogeneous(zero))
    report['localization']={}
    for name,flag in (('learned',False),('zero_shot',True)):
        report['localization'][name]=dict(all=localization(mixed,zero=flag),
            left_right=localization([r for r in mixed if r['condition']=='left_right'],zero=flag),
            quadrants=localization([r for r in mixed if r['condition']=='quadrants'],zero=flag),
            clean_quadrant_fpr=report[name]['quadrants']['clean_false_activation'])
    h=report['learned']['all'];loc=report['localization']['learned']['all']
    precision=h['correct_type_among_all_active_homogeneous_degraded']
    gate=dict(clean_fpr=h['clean_false_activation']<=.15,dark_auc=h['dark']['roc_auc']>=.80,
              bright_auc=h['bright']['roc_auc']>=.80,dark_correct_tpr=h['dark']['tpr_correct_type']>=.40,
              bright_correct_tpr=h['bright']['tpr_correct_type']>=.40,
              active_type_precision=precision is not None and precision>=.85,
              mixed_dark_recall=loc['dark']['correct_activation_recall']>=.40,
              mixed_bright_recall=loc['bright']['correct_activation_recall']>=.40,
              mixed_dark_wrong=loc['dark']['wrong_type_activation_rate']<=.15,
              mixed_bright_wrong=loc['bright']['wrong_type_activation_rate']<=.15)
    report.update(criteria=gate,qualifies_later_pilot=all(gate.values()),failed_criteria=[k for k,v in gate.items() if not v])
    return report


def main():
    parser=argparse.ArgumentParser()
    for key in ('manifest','images','model-identity','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    torch.manual_seed(7);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    args.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(args.manifest.read_text());identity=json.loads(args.model_identity.read_text())
    encoder=FrozenCLIP.from_checkpoint(identity['path'],args.device)
    config=dict(manifest=manifest,model_identity=identity,prompts=PROMPTS,training=TRAIN_CONFIG,
                geometry=encoder.preprocess,conditions=CONDITIONS,device=args.device,dtype='float32',
                zero_shot_calibration='same clean source_calibration IDs/rule; fixed before training')
    (args.output/'config.json').write_text(json.dumps(config,indent=2))
    features=[];records=[]
    for entry in manifest['images']:
        if entry['split']!='source_train':continue
        clean=load_image(args.images/entry['filename']).to(args.device)
        for condition in TRAIN_CONDITIONS:
            with torch.no_grad():
                z=encoder.image_embeddings(degrade(clean,condition)).detach()
            features.append(z)
            records.extend(dict(image_id=entry['image_id'],condition=condition,view=view) for view in VIEW_NAMES)
        print('Encoded source',entry['image_id'],flush=True)
    features=torch.cat(features)
    initial=encoder.text_prototypes.detach().clone()
    init_check=torch.equal(Prototypes(initial).vectors,encoder.text_prototypes)
    trained,history=train_prototypes(initial,features,records,manifest)
    history.update(initialization_exact=init_check,clip_frozen=all(not p.requires_grad for p in encoder.parameters()),
                   clip_grads_absent=all(p.grad is None for p in encoder.parameters()),prototype_parameter_count=trained.vectors.numel())
    (args.output/'training_history.json').write_text(json.dumps(history,indent=2))
    torch.save(dict(features=features.cpu(),records=records),args.output/'source_features.pt')
    weights=args.output/'prototypes.pt'
    torch.save(dict(initial=initial.cpu(),raw=trained.vectors.detach().cpu(),normalized=trained.normalized().detach().cpu()),weights)
    digest=hashlib.sha256(weights.read_bytes()).hexdigest()
    (args.output/'prototype_identity.json').write_text(json.dumps(dict(sha256=digest,bytes=weights.stat().st_size,
          frozen=all(not p.requires_grad for p in trained.parameters()),training_steps=500),indent=2))
    if not history['all_finite']:
        raise RuntimeError('Nonfinite source training; receipts saved. Research-lead guidance required, no evaluation.')
    print('Source training saved/frozen:',history['loss_trajectory'][0],history['loss_trajectory'][-1],digest,flush=True)
    rows=[]
    for entry in manifest['images']:
        if entry['split']!='source_calibration':continue
        image=load_image(args.images/entry['filename']).to(args.device)
        learned,zero=learned_scores(encoder,trained,image)
        items=score_rows(learned.cpu(),image,entry,'clean')
        for row,z in zip(items,zero.cpu()):row.update(zero_d_dark=float(z[0]),zero_d_bright=float(z[1]))
        rows.extend(items)
    calibration=source_calibration(rows,manifest)
    zero_calibration=source_calibration([dict(r,d_dark=r['zero_d_dark'],d_bright=r['zero_d_bright']) for r in rows],manifest)
    (args.output/'calibration.json').write_text(json.dumps(dict(learned=calibration,zero_shot=zero_calibration),indent=2))
    (args.output/'calibration_scores.json').write_text(json.dumps(rows,indent=2))
    pre_eval=dict(prototypes_frozen=all(not p.requires_grad and p.grad is None for p in trained.parameters()),
                  clip_frozen=all(not p.requires_grad and p.grad is None for p in encoder.parameters()))
    (args.output/'pre_evaluation_checks.json').write_text(json.dumps(pre_eval,indent=2))
    assert all(pre_eval.values())
    print('Both clean calibration constants persisted before held-out scores.',flush=True)
    for entry in manifest['images']:
        if entry['split']!='evaluation':continue
        clean=load_image(args.images/entry['filename']).to(args.device)
        for condition in CONDITIONS:
            image=degrade(clean,condition)
            learned,zero=learned_scores(encoder,trained,image)
            # Decisions are computed from score tensors before adding condition/ID metadata.
            lw,la=decisions(learned,calibration);zw,za=decisions(zero,zero_calibration)
            items=score_rows(learned.cpu(),image,entry,condition)
            for i,row in enumerate(items):
                row.update(zero_d_dark=float(zero[i,0]),zero_d_bright=float(zero[i,1]),
                           type=('dark','bright')[lw[i].item()],active=la[i].item(),
                           zero_type=('dark','bright')[zw[i].item()],zero_active=za[i].item())
            rows.extend(items)
        print('Scored held-out',entry['image_id'],flush=True)
    apply_decisions([r for r in rows if r['split']=='source_calibration'],calibration,zero_calibration)
    (args.output/'scores_before_localization.json').write_text(json.dumps(rows,indent=2))
    attach_regions(rows)
    (args.output/'scores.json').write_text(json.dumps(rows,indent=2))
    with (args.output/'scores.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=summarize_prototypes(rows)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    print('T006 audit complete. ISP adaptation not authorized in this task.',flush=True)


if __name__=='__main__':main()
