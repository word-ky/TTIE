"""Execute unchanged official forward AST, excluding target I/O and metrics.

Donor: caiyuanhao1998/Retinexformer, commit
1e9a0efce4b306b6701b824768370ff26066c32a (MIT, Yuanhao Cai 2023).
Network factory is replaced only by direct import of the same RetinexFormer.
"""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import yaml


def load_official(repo, checkpoint, config):
    path = Path(repo) / 'basicsr/models/archs/RetinexFormer_arch.py'
    spec = importlib.util.spec_from_file_location('retinexformer_official_arch', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    network = yaml.safe_load(Path(config).read_text())['network_g'].copy()
    assert network.pop('type') == 'RetinexFormer'
    model = module.RetinexFormer(**network)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['params'], strict=True)
    return torch.nn.DataParallel(model.cuda()).eval()


def official_forward(repo):
    path = Path(repo) / 'Enhancement/test_from_dataset.py'
    tree = ast.parse(path.read_text())
    dataset_branch = next(n for n in tree.body if isinstance(n, ast.If)
                          and ast.unparse(n.test).startswith('dataset in'))
    scope = next(n for n in dataset_branch.orelse if isinstance(n, ast.With))
    loop = next(n for n in scope.body if isinstance(n, ast.For))
    start = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign)
                 and ast.unparse(n.targets[0]) == '(b, c, h, w)')
    end = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.If)
               and ast.unparse(n.test) == 'args.GT_mean')
    body = loop.body[start:end]
    fn = ast.parse('def forward(input_, model_restoration):\n    pass').body[0]
    fn.body = body + [ast.Return(value=ast.Name(id='restored', ctx=ast.Load()))]
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    namespace = {'torch': torch, 'F': F, 'factor': 4,
                 'args': SimpleNamespace(self_ensemble=False)}
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace['forward'], ast.unparse(module)
