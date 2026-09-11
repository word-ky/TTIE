"""T006: source-only training of three normalized semantic vectors."""

import torch
from torch import nn
from torch.nn import functional as F

from .clip_signal import calibrate


TRAIN_CONDITIONS = ('clean', 'homogeneous_dark', 'homogeneous_bright')
TRAIN_CONFIG = dict(optimizer='AdamW', lr=.005, weight_decay=.0001, steps=500,
                    temperature=.07, seed=7, batch='full', class_weighting='equal')


class Prototypes(nn.Module):
    def __init__(self, initial):
        super().__init__()
        self.vectors = nn.Parameter(initial.detach().clone())

    def normalized(self):
        return F.normalize(self.vectors, dim=-1)

    def forward(self, embeddings):
        return embeddings @ self.normalized().T

    def scores(self, embeddings):
        similarities = self(embeddings)
        return similarities[:,1:] - similarities[:,:1]


def train_prototypes(initial, features, records, manifest, *, steps=500):
    source_ids = {r['image_id'] for r in manifest['images'] if r['split']=='source_train'}
    selected = [i for i,r in enumerate(records) if r['image_id'] in source_ids and r['condition'] in TRAIN_CONDITIONS]
    embeddings = features[selected].detach()
    labels = torch.tensor([TRAIN_CONDITIONS.index(records[i]['condition']) for i in selected], device=embeddings.device)
    torch.manual_seed(7)
    model = Prototypes(initial)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.005, weight_decay=.0001)
    losses, gradients = [], []
    finite = True
    for step in range(steps+1):
        loss = F.cross_entropy(model(embeddings)/.07, labels)
        losses.append(loss.detach().item())
        finite = bool(torch.isfinite(loss)) and bool(torch.isfinite(model.vectors).all())
        if not finite or step==steps: break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = model.vectors.grad.norm()
        gradients.append(norm.item())
        if not torch.isfinite(norm):
            finite=False;break
        optimizer.step()
    model.eval().requires_grad_(False)
    initial_unit=F.normalize(initial.detach(),dim=-1)
    final=model.normalized().detach()
    history=dict(config={**TRAIN_CONFIG,'steps':steps},loss_trajectory=losses,gradient_norms=gradients,
                 all_finite=finite,source_ids=sorted(source_ids),example_count=len(selected),
                 class_counts=torch.bincount(labels,minlength=3).tolist(),
                 initial_cosines=(initial_unit@initial_unit.T).cpu().tolist(),
                 final_cosines=(final@final.T).cpu().tolist(),
                 initial_final_alignment=(initial_unit*final).sum(dim=-1).cpu().tolist(),
                 frozen_after_training=all(not p.requires_grad for p in model.parameters()))
    optimizer.zero_grad(set_to_none=True)
    return model, history


def source_calibration(rows, manifest):
    mapped={'images':[dict(r,split='calibration' if r['split']=='source_calibration' else r['split']) for r in manifest['images']]}
    return calibrate(rows,mapped)


def learned_scores(encoder, prototypes, image):
    """No IDs, conditions, references or masks accepted at inference."""
    with torch.no_grad():
        z=encoder.image_embeddings(image).detach()
        learned=prototypes.scores(z)
        zero_sim=z@encoder.text_prototypes.T
        return learned.detach(), (zero_sim[:,1:]-zero_sim[:,:1]).detach()
