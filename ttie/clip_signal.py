"""Frozen OpenCLIP scores: image-only API, fixed prompts and calibration."""

import torch
from torch import nn
from torch.nn import functional as F

from .natural import clip_pixels


PROMPTS = {
    'normal': ['a well-exposed natural photograph', 'a normally exposed clear photo',
               'a photo with natural brightness and contrast'],
    'dark': ['a dark underexposed photograph', 'a low-light photo with poor visibility',
             'an underexposed dark image'],
    'bright': ['an overexposed washed-out photograph', 'a photo with blown highlights',
               'an excessively bright overexposed image'],
}


class FrozenCLIP(nn.Module):
    def __init__(self, model, text_prototypes, *, size=224, mean=None, std=None):
        super().__init__()
        self.model = model.eval().requires_grad_(False)
        self.register_buffer('text_prototypes', text_prototypes.detach())
        self.preprocess = dict(size=size)
        if mean is not None:
            self.preprocess.update(mean=mean, std=std)

    @classmethod
    def from_checkpoint(cls, checkpoint, device='cuda:0'):
        import open_clip
        model = open_clip.create_model('ViT-B-32', pretrained=checkpoint, device=device, precision='fp32')
        model.eval().requires_grad_(False)
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        prototypes = []
        with torch.no_grad():
            for ensemble in PROMPTS.values():
                embeddings = F.normalize(model.encode_text(tokenizer(ensemble).to(device)), dim=-1)
                prototypes.append(F.normalize(embeddings.mean(dim=0), dim=0))
        cfg = open_clip.get_pretrained_cfg('ViT-B-32', 'laion2b_s34b_b79k')
        return cls(model, torch.stack(prototypes), mean=cfg['mean'], std=cfg['std'])

    def image_embeddings(self, image):
        return F.normalize(self.model.encode_image(clip_pixels(image, **self.preprocess)), dim=-1)

    def forward(self, image):
        embedding = self.image_embeddings(image)
        similarities = embedding @ self.text_prototypes.T
        return similarities[:, 1:] - similarities[:, :1]


def calibrate(rows, manifest):
    """Use only clean rows from calibration IDs; ignore all held-out values."""
    ids = {r['image_id'] for r in manifest['images'] if r['split'] == 'calibration'}
    selected = [r for r in rows if r['image_id'] in ids and r['condition'] == 'clean']
    values = torch.tensor([[r['d_dark'], r['d_bright']] for r in selected], dtype=torch.float64)
    return dict(tau=torch.quantile(values, .95, dim=0, interpolation='linear').tolist(),
                scale=values.std(dim=0, correction=0).clamp_min(.01).tolist(),
                image_ids=sorted(ids), view_count=len(selected), percentile=95,
                std_correction=0, scale_floor=.01)


def decisions(scores, calibration):
    """Detached original-input winner/type; no metadata accepted."""
    values = scores.detach()
    winner = values.argmax(dim=-1)
    tau = values.new_tensor(calibration['tau'])
    active = values.gather(1, winner[:, None]).squeeze(1) > tau[winner]
    return winner, active


def gated_objective(scorer, original, calibration):
    """Fixed input gate; reused only if the Stage-A gate authorizes Stage B."""
    with torch.no_grad():
        winner, active = decisions(scorer(original), calibration)

    def loss(output):
        if not active.any():
            return output.sum() * 0
        scores = scorer(output)
        k = winner[active]
        selected = scores[active].gather(1, k[:, None]).squeeze(1)
        tau = scores.new_tensor(calibration['tau'])[k]
        scale = scores.new_tensor(calibration['scale'])[k]
        return F.softplus((selected-tau)/scale).mean()

    return loss, winner, active
