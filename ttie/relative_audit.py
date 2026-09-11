"""Fresh T005 relative-vs-absolute signal audit; no adaptation is launched."""

import argparse
import csv
import json
from pathlib import Path

import torch

from .clip_audit import audit_group, score_rows
from .clip_signal import FrozenCLIP, PROMPTS, decisions
from .natural import CONDITIONS, degrade, load_image, views
from .relative_clip import DELTA_EV, probes, relative_scores, relative_calibration, relative_gate
from .suite import clipping_fraction


def relative_rows(scorer, image, entry, condition):
    scores = {key: value.cpu() for key, value in relative_scores(scorer, image).items()}
    rows = score_rows(scores['original'], image, entry, condition)
    plus, minus = probes(image)
    for i, (row, p, m) in enumerate(zip(rows, views(plus), views(minus))):
        row.update(plus_d_dark=float(scores['plus'][i, 0]), plus_d_bright=float(scores['plus'][i, 1]),
                   minus_d_dark=float(scores['minus'][i, 0]), minus_d_bright=float(scores['minus'][i, 1]),
                   r_dark=float(scores['response'][i, 0]), r_bright=float(scores['response'][i, 1]),
                   plus_clipping_fraction=clipping_fraction(p), minus_clipping_fraction=clipping_fraction(m))
    return rows


def attach_decisions(rows, calibration, absolute_calibration):
    for row in rows:
        r = torch.tensor([[row['r_dark'], row['r_bright']]])
        targets = torch.tensor([[row['plus_d_dark'], row['minus_d_bright']]])
        winner, active, target = relative_gate(dict(response=r, targets=targets), calibration)
        absolute_winner, absolute_active = decisions(torch.tensor([[row['d_dark'], row['d_bright']]]), absolute_calibration)
        row.update(type=('dark','bright')[winner.item()], active=active.item(), target=target.item(),
                   absolute_type=('dark','bright')[absolute_winner.item()], absolute_active=absolute_active.item())


def summarize_relative(rows):
    evaluation = [r for r in rows if r['split']=='evaluation']
    relative = [dict(r, d_dark=r['r_dark'], d_bright=r['r_bright']) for r in evaluation]
    absolute = [dict(r, type=r['absolute_type'], active=r['absolute_active']) for r in evaluation]
    def groups(items):
        return dict(all=audit_group(items), full=audit_group([r for r in items if r['view']=='full']),
                    quadrants=audit_group([r for r in items if r['view']!='full']))
    result = dict(heldout_rows=len(evaluation), relative=groups(relative), absolute=groups(absolute))
    all_views = result['relative']['all']
    precision = all_views['correct_type_among_all_active_homogeneous_degraded']
    criteria = dict(clean_fpr=all_views['clean_false_activation'] <= .15,
                    dark_auc=all_views['dark']['roc_auc'] >= .75,
                    bright_auc=all_views['bright']['roc_auc'] >= .75,
                    dark_correct_tpr=all_views['dark']['tpr_correct_type'] >= .30,
                    bright_correct_tpr=all_views['bright']['tpr_correct_type'] >= .30,
                    active_type_precision=precision is not None and precision >= .80)
    result.update(criteria=criteria, stage_a_pass=all(criteria.values()), failed_criteria=[k for k,v in criteria.items() if not v])
    return result


def main():
    parser = argparse.ArgumentParser()
    for option in ('manifest','images','model-identity','absolute-calibration','output'):
        parser.add_argument('--'+option, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.manual_seed(7); torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    identity = json.loads(args.model_identity.read_text())
    absolute_calibration = json.loads(args.absolute_calibration.read_text())
    scorer = FrozenCLIP.from_checkpoint(identity['path'], args.device)
    config = dict(manifest=manifest, model_identity=identity, prompts=PROMPTS, delta_EV=DELTA_EV,
                  geometry=scorer.preprocess, conditions=CONDITIONS, dtype='float32', device=args.device,
                  absolute_calibration=absolute_calibration, seed=7,
                  model_frozen=all(not p.requires_grad for p in scorer.parameters()))
    (args.output/'config.json').write_text(json.dumps(config, indent=2))
    rows = []
    for entry in manifest['images']:
        if entry['split']=='calibration':
            image = load_image(args.images/entry['filename']).to(args.device)
            rows.extend(relative_rows(scorer, image, entry, 'clean'))
    calibration = relative_calibration(rows, manifest)
    (args.output/'calibration.json').write_text(json.dumps(calibration, indent=2))
    (args.output/'calibration_scores.json').write_text(json.dumps(rows, indent=2))
    print('Relative calibration frozen:', calibration, flush=True)
    for entry in manifest['images']:
        if entry['split']!='evaluation': continue
        clean = load_image(args.images/entry['filename']).to(args.device)
        for condition in CONDITIONS:
            image = degrade(clean, condition)
            rows.extend(relative_rows(scorer, image, entry, condition))
        print('Scored fresh held-out image', entry['image_id'], flush=True)
    attach_decisions(rows, calibration, absolute_calibration)
    (args.output/'scores.json').write_text(json.dumps(rows, indent=2))
    with (args.output/'scores.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = summarize_relative(rows)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print('Stage B authorized by gate:', summary['stage_a_pass'], flush=True)


if __name__=='__main__':
    main()
