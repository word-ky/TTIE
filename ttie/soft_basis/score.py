"""Label-free scoring process. No reference files or evaluation metadata inputs."""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import torch
from ..clip_signal import FrozenCLIP
from ..learned_prototypes import Prototypes
from ..semantic_ttt import SemanticScorer
from ..energy_model import load_energy, SCHEMA
from ..sobolev_receipt import verify_receipt
from ..stop_receipt import sha
from ..routing.provenance import T015_SOURCES, verify_source
from .selection import BOUNDARIES, score_episode

SOURCES = T015_SOURCES + tuple('ttie/soft_basis/'+name+'.py' for name in
    ('__init__', 'renderer', 'selection', 'score', 'prepare', 'evaluate')) + ('scripts/run_t016b_a6000.sh',)


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path, value): Path(path).write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')
def tensor_sha(t): return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def score_inputs(inputs, scorer, head, calibration, device):
    index = read(inputs/'inputs.json')
    rows = []
    for entry in index['episodes']:
        path = inputs/entry['file']
        assert sha(path) == entry['sha256']
        data = torch.load(path, weights_only=True, map_location='cpu')
        image, corners = data['image'].to(device), data['corners'].to(device)
        result = score_episode(image, corners, data['gate'], calibration, scorer, head)
        rows.append(dict(episode=entry['episode'], input_file_sha256=entry['sha256'],
                         pixels_sha256=tensor_sha(image), corners_sha256=tensor_sha(corners),
                         gate=data['gate'], **result))
        print('T016-B scored', len(rows), entry['episode'], flush=True)
    return rows


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    for key in ('inputs', 'source-manifest', 'model-identity', 'prototypes', 'receipt',
                'energy', 'control', 'energy-receipt', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--source-sha', required=True); p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    code = verify_source(a.source_sha, SOURCES)
    torch.manual_seed(7); torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    gate_receipt, identity = read(a.receipt), read(a.model_identity)
    frozen = verify_receipt(a.energy_receipt, a.energy, a.control, a.source_manifest, identity, gate_receipt)
    assert sha(a.prototypes) == gate_receipt['prototype_identity']['sha256']
    assert sha(Path(identity['path'])) == gate_receipt['model_identity']['sha256']
    encoder = FrozenCLIP.from_checkpoint(identity['path'], a.device)
    saved = torch.load(a.prototypes, map_location=a.device, weights_only=True)
    prototypes = Prototypes(saved['raw']); scorer = SemanticScorer(encoder, prototypes)
    head = load_energy(a.energy).to(a.device).eval().requires_grad_(False)
    assets = {key: sha(getattr(a, key.replace('-', '_'))) for key in
              ('source-manifest', 'model-identity', 'prototypes', 'receipt', 'energy', 'control', 'energy-receipt')}
    assets['clip_checkpoint'] = sha(Path(identity['path']))
    config = dict(task='T016-B', source_sha=a.source_sha, source_code_sha256=code,
                  assets_sha256=assets, input_index_sha256=sha(a.inputs/'inputs.json'),
                  frozen_t014_source_sha=frozen['source_sha'], schema=SCHEMA,
                  normalization=head.normalization(), candidates=BOUNDARIES,
                  seed=7, tf32=False, device=a.device, optimization_steps=0, new_image_ids=0,
                  selection_inputs='persisted degraded pixels, fixed corners, original gate constants, frozen assets')
    a.output.mkdir(parents=True, exist_ok=True)
    write(a.output/'config.json', config)
    rows = score_inputs(a.inputs, scorer, head, gate_receipt['calibration'], a.device)
    assert len(rows) == 120
    assert all(not q.requires_grad and q.grad is None for q in head.parameters())
    assert all(not q.requires_grad and q.grad is None for q in scorer.parameters())
    assert sha(a.energy) == assets['energy'] and torch.equal(prototypes.vectors, saved['raw'])
    write(a.output/'selection.json', dict(candidates=BOUNDARIES, episodes=rows))
    write(a.output/'selection_receipt.json', dict(finalized_utc=datetime.now(timezone.utc).isoformat(),
          selection_sha256=sha(a.output/'selection.json'), config_sha256=sha(a.output/'config.json'),
          episodes=len(rows), energies=len(rows)*9, frozen_assets_unchanged=True,
          reference_access=False, source_sha=a.source_sha))
    print('T016-B selection finalized', sha(a.output/'selection.json'), flush=True)


if __name__ == '__main__': main()
