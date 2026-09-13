"""Frozen T018-D selector: five 28-D feature vectors, no reference inputs."""
import hashlib
import json
from pathlib import Path
import torch
from .direction_probe import CLASSES, RECIPE, DirectionHead, predict

INFERENCE_SOURCES = ('ttie/__init__.py', 'ttie/direction_probe.py', 'ttie/direction_selector.py')
CROSS_NAMES = ('center', 'x_lower', 'x_upper', 'y_lower', 'y_upper')
HARD = tuple((x, y, 0.) for x in (.4, .5, .6) for y in (.4, .5, .6))


def cross_features(center, x_lower, x_upper, y_lower, y_upper):
    """Accept a single cross or a batch; return two [N,84] float32 tensors."""
    f = [torch.atleast_2d(torch.as_tensor(v, dtype=torch.float32, device='cpu'))
         for v in (center, x_lower, x_upper, y_lower, y_upper)]
    assert all(v.ndim == 2 and v.shape == f[0].shape and v.shape[1] == 28 for v in f)
    return tuple(torch.cat((f[0], f[lo] - f[0], f[hi] - f[0]), dim=1) for lo, hi in ((1, 2), (3, 4)))


class DirectionSelector:
    def __init__(self, head_x, head_y):
        torch.set_num_threads(1)
        self.head_x = head_x.cpu().eval().requires_grad_(False)
        self.head_y = head_y.cpu().eval().requires_grad_(False)

    def predict(self, center, x_lower, x_upper, y_lower, y_upper):
        """Return one logits/classes/coordinate/index record per input row.

        score_index addresses the nine hard candidates; hard_index addresses
        the inherited 27-candidate table (tau=0 entries, stride three).
        """
        zx, zy = cross_features(center, x_lower, x_upper, y_lower, y_upper)
        lx, cx = predict(self.head_x, zx); ly, cy = predict(self.head_y, zy)
        result = []
        for ax, ay, ix, iy in zip(lx, ly, cx, cy):
            bx, by = CLASSES[ix], CLASSES[iy]; index = HARD.index((bx, by, 0.))
            result.append(dict(x_logits=ax, y_logits=ay, x_class=ix, y_class=iy,
                               bx=bx, by=by, score_index=index, hard_index=3 * index))
        return result


def load_selector(folder, expected_receipt_sha256):
    """Load only the pinned receipt, two heads, and current inference source bytes."""
    folder = Path(folder); raw = (folder / 'selector_frozen.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_receipt_sha256:
        raise ValueError('Selector receipt hash mismatch')
    receipt = json.loads(raw); assert receipt['recipe'] == RECIPE
    source_root = Path(__file__).resolve().parents[1]
    for path in INFERENCE_SOURCES:
        if hashlib.sha256((source_root / path).read_bytes()).hexdigest() != receipt['inference_code_sha256'][path]:
            raise ValueError('Inference code hash mismatch: ' + path)
    heads = []
    for axis in ('x', 'y'):
        path = folder / ('head_' + axis + '.pt')
        if hashlib.sha256(path.read_bytes()).hexdigest() != receipt['files_sha256'][path.name]:
            raise ValueError('Head hash mismatch: ' + axis)
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        assert checkpoint['recipe'] == RECIPE
        head = DirectionHead(); head.load_state_dict(checkpoint['state_dict']); heads.append(head)
    return DirectionSelector(*heads)
