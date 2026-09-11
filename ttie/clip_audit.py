"""Offline signal-quality metrics; no calibration or adaptation tuning."""

import argparse
import csv
import json
from pathlib import Path

import torch

from .clip_signal import FrozenCLIP, PROMPTS, calibrate, decisions
from .natural import CONDITIONS, VIEW_NAMES, degrade, load_image, views
from .suite import clipping_fraction


def auc(positives, negatives):
    return sum(float(p > n) + .5*float(p == n) for p in positives for n in negatives) / (len(positives)*len(negatives))


def audit_group(rows):
    clean = [r for r in rows if r['condition'] == 'clean']
    by_pair = {(r['image_id'], r['view']): r for r in clean}
    result = dict(clean_view_count=len(clean), clean_false_activation=sum(r['active'] for r in clean)/len(clean))
    active_degraded = []
    for kind, condition in (('dark', 'homogeneous_dark'), ('bright', 'homogeneous_bright')):
        degraded = [r for r in rows if r['condition'] == condition]
        active = [r for r in degraded if r['active']]
        key = 'd_'+kind
        correct = sum(r['type'] == kind for r in active)
        result[kind] = dict(view_count=len(degraded), tpr_any_activation=len(active)/len(degraded),
                            tpr_correct_type=correct/len(degraded),
                            correct_type_among_active=correct/len(active) if active else None,
                            roc_auc=auc([r[key] for r in degraded], [r[key] for r in clean]),
                            paired_increase=sum(r[key] > by_pair[r['image_id'],r['view']][key] for r in degraded)/len(degraded))
        active_degraded.extend((r['type'] == kind) for r in active)
    result['correct_type_among_all_active_homogeneous_degraded'] = sum(active_degraded)/len(active_degraded) if active_degraded else None
    return result


def summarize(rows):
    heldout = [r for r in rows if r['split'] == 'evaluation']
    groups = dict(all=audit_group(heldout),
                  full=audit_group([r for r in heldout if r['view']=='full']),
                  quadrants=audit_group([r for r in heldout if r['view']!='full']))
    a = groups['all']
    criteria = dict(clean_fpr=a['clean_false_activation'] <= .15,
                    dark_auc=a['dark']['roc_auc'] >= .75, bright_auc=a['bright']['roc_auc'] >= .75,
                    dark_paired=a['dark']['paired_increase'] >= .75, bright_paired=a['bright']['paired_increase'] >= .75)
    return dict(heldout_rows=len(heldout), groups=groups, criteria=criteria,
                stage_a_pass=all(criteria.values()), failed_criteria=[k for k,v in criteria.items() if not v])


def image_scores(scorer, image):
    """Pixel-only scoring seam. No clean target, condition, label or mask."""
    with torch.no_grad():
        return scorer(image).detach().cpu()


def score_rows(scores, image, entry, condition):
    # Attach metadata ONLY after scoring; callers can change it without affecting scores.
    return [dict(image_id=entry['image_id'], split=entry['split'], condition=condition,
                 view=name, d_dark=float(pair[0]), d_bright=float(pair[1]),
                 input_clipping_fraction=clipping_fraction(crop))
            for name, pair, crop in zip(VIEW_NAMES, scores, views(image))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--images', required=True, type=Path)
    parser.add_argument('--model-identity', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.manual_seed(7)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    identity = json.loads(args.model_identity.read_text())
    scorer = FrozenCLIP.from_checkpoint(identity['path'], args.device)
    config = dict(model_identity=identity, prompts=PROMPTS, manifest=manifest, device=args.device,
                  geometry=scorer.preprocess, conditions=CONDITIONS, seed=7, dtype='float32',
                  stage_b_optimizer=dict(name='Adam', steps=40, lr=.01))
    (args.output/'config.json').write_text(json.dumps(config, indent=2))
    rows = []
    for entry in manifest['images']:
        if entry['split'] != 'calibration':
            continue
        image = load_image(args.images/entry['filename']).to(args.device)
        rows.extend(score_rows(image_scores(scorer, image), image, entry, 'clean'))
    calibration = calibrate(rows, manifest)
    (args.output/'calibration.json').write_text(json.dumps(calibration, indent=2))
    # Flush frozen constants to disk before any held-out score is computed.
    print('Calibration frozen:', calibration, flush=True)
    first = next(r for r in manifest['images'] if r['split']=='calibration')
    image = load_image(args.images/first['filename']).to(args.device).requires_grad_()
    score = scorer(image)
    # PyTorch 2.4 has no deterministic CUDA antialiased-bicubic backward.
    # Permit that operation for this calibration-only gradient check, measure
    # repeat sensitivity, then restore strict mode for all held-out scoring.
    torch.use_deterministic_algorithms(False)
    gradient, = torch.autograd.grad(score[:,0].mean(), image, retain_graph=True)
    repeat_gradient, = torch.autograd.grad(score[:,0].mean(), image)
    torch.use_deterministic_algorithms(True)
    gradient_check = dict(frozen=all(not p.requires_grad for p in scorer.parameters()),
                          model_grads_absent=all(p.grad is None for p in scorer.parameters()),
                          input_gradient_finite=bool(torch.isfinite(gradient).all()),
                          input_gradient_norm=gradient.norm().item(),
                          repeat_gradient_max_difference=(gradient-repeat_gradient).abs().max().item(),
                          gradient_check_strict_determinism=False,
                          repeat_score_max_difference=(score.detach().cpu()-image_scores(scorer,image)).abs().max().item())
    assert gradient_check['frozen'] and gradient_check['model_grads_absent']
    assert gradient_check['input_gradient_finite'] and gradient_check['input_gradient_norm'] > 0
    (args.output/'real_clip_gradient_check.json').write_text(json.dumps(gradient_check, indent=2))
    del gradient, repeat_gradient, score, image
    for entry in manifest['images']:
        if entry['split'] != 'evaluation':
            continue
        clean = load_image(args.images/entry['filename']).to(args.device)
        for condition in CONDITIONS:
            current = degrade(clean, condition)
            scores = image_scores(scorer, current)
            rows.extend(score_rows(scores, current, entry, condition))
        print('Scored held-out image', entry['image_id'], flush=True)
    for row in rows:
        winner, active = decisions(torch.tensor([[row['d_dark'], row['d_bright']]]), calibration)
        row.update(type=('dark','bright')[winner.item()], active=active.item())
    (args.output/'scores.json').write_text(json.dumps(rows, indent=2))
    with (args.output/'scores.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    report = summarize(rows)
    (args.output/'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print('Stage B authorized by gate:', report['stage_a_pass'], flush=True)


if __name__ == '__main__':
    main()
