"""T016-B: fixed hard renderers scored through the unchanged T014 feature/head API."""
from types import SimpleNamespace
import torch
from ..energy_model import features
from .renderer import CANDIDATES, FixedCorners

HARD_INDICES = tuple(i for i, c in enumerate(CANDIDATES) if c[2] == 0)
BOUNDARIES = tuple(CANDIDATES[i] for i in HARD_INDICES)


def choose(energies):
    order = sorted(range(9), key=lambda i: (energies[i], i))
    return dict(selected_index=order[0], margin=energies[order[1]]-energies[order[0]],
                minimum_ties=sum(e == energies[order[0]] for e in energies))


@torch.no_grad()
def score_episode(image, corners, gate, calibration, scorer, head):
    objective = SimpleNamespace(
        active=torch.tensor(gate['active'], device=image.device, dtype=torch.bool),
        winner=torch.tensor(gate['winner'], device=image.device, dtype=torch.long),
        evidence=torch.tensor(gate['evidence'], device=image.device, dtype=torch.float64),
        calibration=calibration)
    energies, vectors, scores = [], [], []
    for candidate in BOUNDARIES:
        model = FixedCorners(corners, candidate).to(image)
        current = scorer(model(image))
        vector = features(objective, current, corners)
        value = head(vector).squeeze()
        assert torch.isfinite(value)
        assert torch.equal(model.corners, corners)
        energies.append(float(value)); vectors.append(vector.cpu().tolist())
        scores.append(current.cpu().tolist())
    return dict(energies=energies, features=vectors, clip_scores=scores, **choose(energies))
