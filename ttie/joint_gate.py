"""T007: one clean image-level threshold; the T006 scores/type stay frozen."""
import torch

from .clip_signal import decisions
from .natural import VIEW_NAMES


def winning_evidence(scores, calibration):
    winner, baseline = decisions(scores, calibration)
    values = scores.detach().to(torch.float64)
    tau = values.new_tensor(calibration['tau'])
    scale = values.new_tensor(calibration['scale'])
    evidence = (values.gather(1, winner[:, None]).squeeze(1)-tau[winner])/scale[winner]
    return winner, baseline, evidence


def joint_calibration(rows, manifest, calibration):
    ids = sorted(r['image_id'] for r in manifest['images'] if r['split']=='source_calibration')
    maxima = []
    for image_id in ids:
        group = [r for r in rows if r['image_id']==image_id and r['condition']=='clean']
        assert sorted(r['view'] for r in group)==sorted(VIEW_NAMES)
        scores = torch.tensor([[r['d_dark'], r['d_bright']] for r in group], dtype=torch.float32)
        maxima.append(winning_evidence(scores, calibration)[2].max().item())
    q = torch.quantile(torch.tensor(maxima, dtype=torch.float64), .95, interpolation='linear').item()
    return dict(q_joint=q, image_ids=ids, image_maxima=maxima, percentile=95,
                interpolation='linear', evidence_dtype='float64', calibration=calibration)


def both_gates(scores, receipt):
    winner, baseline, evidence = winning_evidence(scores, receipt['calibration'])
    return winner, baseline, evidence > receipt['q_joint'], evidence
