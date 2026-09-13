"""Evaluation-only replay of the accepted T014 frozen outputs for T021-A."""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import torch

from .ssim_transfer import METRIC, BOOTSTRAP, rgb_ssim, cluster_bootstrap

H0 = 'region2_ttt_energy_value_only'
H1 = 'region2_ttt_energy_sobolev'
CONDITIONS = ['clean', 'homogeneous_dark', 'homogeneous_bright', 'left_right', 'quadrants']
CLARIFICATION = 'd7dc7ec8c2fa4599b946d1860a7453b43f730387'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def select_rows(receipt):
    rows = [r for r in receipt['rows'] if r['condition'] in CONDITIONS]
    assert len(rows) == 200
    ids = sorted({r['image_id'] for r in rows})
    assert len(ids) == 40
    assert all(sorted(r['condition'] for r in rows if r['image_id'] == i) == sorted(CONDITIONS) for i in ids)
    clean = {r['image_id']: r for r in rows if r['condition'] == 'clean'}
    return rows, clean


def array(tensor):
    return tensor.squeeze(0).permute(1, 2, 0).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    receipt = json.loads(args.receipt.read_bytes())
    rows, clean = select_rows(receipt)
    # All selected byte identities are checked before any reference tensor is read.
    for row in rows:
        for file in row['files'].values():
            assert Path(file['path']).stat().st_size == file['bytes']
            assert sha(file['path']) == file['sha256']
    binding = dict(clarification_commit=CLARIFICATION, H0=H0, H1=H1,
                   conditions=CONDITIONS, metric=METRIC, bootstrap=BOOTSTRAP,
                   receipt_sha256=sha(args.receipt), accepted_origins=receipt['accepted_origins'],
                   metric_source_sha256=sha(Path(__file__).with_name('ssim_transfer.py')),
                   audit_source_sha256=sha(__file__),
                   frozen_before_tensor_read_utc=datetime.now(timezone.utc).isoformat(),
                   versions=dict(numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__),
                   reference='Same-source clean-condition frozen identity.image; no reload, resize or rendering',
                   rows=rows)
    dump(args.out / 'binding.json', binding)
    references = {i: array(torch.load(r['files']['outputs.pt']['path'], map_location='cpu', weights_only=True)['identity']['image'])
                  for i, r in clean.items()}
    table = []
    for row in rows:
        outputs = torch.load(row['files']['outputs.pt']['path'], map_location='cpu', weights_only=True)
        reference = references[row['image_id']]
        s0 = rgb_ssim(array(outputs[H0]['image']), reference)
        s1 = rgb_ssim(array(outputs[H1]['image']), reference)
        table.append(dict(image_id=row['image_id'], condition=row['condition'], SSIM0=s0, SSIM1=s1,
                          delta=s1-s0, output_path=row['files']['outputs.pt']['path'],
                          output_sha256=row['files']['outputs.pt']['sha256'],
                          reference_path=clean[row['image_id']]['files']['outputs.pt']['path'],
                          reference_sha256=clean[row['image_id']]['files']['outputs.pt']['sha256']))
        if len(table) % 40 == 0:
            print(f'Completed {len(table)}/200 rows', flush=True)
    with (args.out / 'ssim_rows.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader(); writer.writerows(table)
    stats, means, draws = cluster_bootstrap([r['image_id'] for r in table], [r['delta'] for r in table])
    stats['conditions'] = {c: dict(rows=40, mean=float(np.mean([r['delta'] for r in table if r['condition'] == c]))) for c in CONDITIONS}
    stats['verdict'] = 'positive' if stats['positive'] else 'negative'
    np.save(args.out / 'bootstrap_means.npy', means)
    np.save(args.out / 'bootstrap_draws.npy', draws)
    dump(args.out / 'summary.json', stats)
    dump(args.out / 'artifact_hashes.json', {p.name: sha(p) for p in sorted(args.out.iterdir()) if p.is_file()})
    print(json.dumps(stats), flush=True)


if __name__ == '__main__':
    main()
