"""TTIE-authored independent adapter calling external pinned official methods.

No third-party source is embedded or generated into repository artifacts.
Native blur first, reflect-pad low and blur, then official direct test (not test4).
"""
import ast
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import torch
import torch.nn.functional as F
import yaml


def load_adapter(root):
    root = Path(root)
    sys.path.insert(0, str(root / 'official'))
    from models import networks
    from models.Video_base_model4_m import VideoBaseModel
    from data.util import read_img_seq
    opt = yaml.safe_load((root / 'official/options/test/LOLv2_real.yml').read_text())
    # Official parse converts missing keys to None; the network's center default.
    opt['network_G']['center'] = None
    model = networks.define_G(opt)
    model.load_state_dict(torch.load(root / 'LOLv2_real.pth', map_location='cpu', weights_only=True), strict=True)
    state = SimpleNamespace(netG=torch.nn.DataParallel(model.cuda()).eval())
    path = root / 'official/data/dataset_LOLv2_real.py'
    tree = ast.parse(path.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '__getitem__')
    body = [n for n in method.body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'img_nf' for t in n.targets)]
    assert len(body) == 4
    fn = ast.parse('def blur(img_LQ):\n    pass').body[0]
    fn.body = body + [ast.Return(value=ast.Name(id='img_nf', ctx=ast.Load()))]
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    namespace = {'torch': torch, 'cv2': cv2}
    exec(compile(module, str(path), 'exec'), namespace)

    def forward(image):
        feature = namespace['blur'](image)
        height, width = image.shape[-2:]
        pad = (0, (16 - width % 16) % 16, 0, (16 - height % 16) % 16)
        state.var_L = F.pad(image.unsqueeze(0), pad, mode='reflect').cuda()
        state.nf = F.pad(feature.unsqueeze(0), pad, mode='reflect').cuda()
        VideoBaseModel.test(state)
        result = state.fake_H[:, :, :height, :width].clamp(0, 1)
        return result[0].detach().cpu().permute(1, 2, 0).numpy()

    return state.netG, forward, read_img_seq


def parameter_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
