"""T005 fixed exposure probes and pixel-only relative evidence."""

import torch

from .clip_signal import calibrate, decisions


DELTA_EV = .25


def probes(image):
    return (image * 2**DELTA_EV).clamp(0, 1), (image * 2**(-DELTA_EV)).clamp(0, 1)


def relative_scores(scorer, image):
    """All counterfactuals derive only from the current input pixels."""
    plus, minus = probes(image)
    with torch.no_grad():
        original = scorer(image).detach()
        positive = scorer(plus).detach()
        negative = scorer(minus).detach()
    response = torch.stack((original[:, 0]-positive[:, 0], original[:, 1]-negative[:, 1]), dim=1)
    targets = torch.stack((positive[:, 0], negative[:, 1]), dim=1)
    return dict(original=original, plus=positive, minus=negative, response=response, targets=targets)


def relative_calibration(rows, manifest):
    mapped = [dict(r, d_dark=r['r_dark'], d_bright=r['r_bright']) for r in rows]
    return calibrate(mapped, manifest)


def relative_gate(scores, calibration):
    winner, active = decisions(scores['response'], calibration)
    target = scores['targets'].gather(1, winner[:, None]).squeeze(1).detach()
    return winner, active, target
