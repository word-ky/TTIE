"""T016-C CPU development probes: fixed image-grouped folds, no deployable selector."""
import torch
from .stop_quality import QualityHead, train_head


def grouped_folds(metadata):
    ids = sorted({r['image_id'] for r in metadata})
    assigned = {image_id: j % 5 for j, image_id in enumerate(ids)}
    return [dict(fold=f, train=[i for i, r in enumerate(metadata) if assigned[r['image_id']] != f],
                 heldout=[i for i, r in enumerate(metadata) if assigned[r['image_id']] == f],
                 train_image_ids=[i for i in ids if assigned[i] != f],
                 heldout_image_ids=[i for i in ids if assigned[i] == f]) for f in range(5)]


def probe_features(saved, candidates, dimension):
    x = torch.tensor(saved, dtype=torch.float32)
    assert x.shape[-2:] == (9, 28)
    if dimension == 28: return x
    assert dimension == 30
    geometry = x.new_tensor([((c[0]-.5)/.1, (c[1]-.5)/.1) for c in candidates])
    return torch.cat((x, geometry.expand(*x.shape[:-2], 9, 2)), dim=-1)


def train_probe(x, mse):
    # Reuse the exact T014 value-head optimizer, normalization and final epoch recipe.
    return train_head(x, mse, head_factory=lambda: QualityHead(x.shape[-1], torch.nn.SiLU))


@torch.no_grad()
def predict(head, heldout_features):
    values = head(heldout_features.reshape(-1, heldout_features.shape[-1])).reshape(-1, 9)
    assert torch.isfinite(values).all()
    # Rows are independent candidate values; torch.argmin returns the first exact minimum.
    return values.tolist(), values.argmin(dim=1).tolist()
