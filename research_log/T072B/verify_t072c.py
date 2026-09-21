"""T072-C read-only audit of three native UHD-LL smoke receipts."""
import argparse, hashlib, json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from research_log.T063A.common import sha, thash, utc, write


METHODS = ['ours', 'retinexformer', 'snr_aware']
OLD = Path('/media/wenchang/F/wjq/TTIE/runs/T072B-uhdll-native/ours')


def main(low, out):
    here = Path('research_log/T072B')
    get = lambda name: json.loads((here / name).read_bytes())
    pairs, choice, acquired = get('pairs_manifest.json'), get('smoke_selection.json'), get('input_receipt.json')
    inputs = {row['name']: row for row in get('input_metadata.json')['files']}
    gts = {row['name']: row for row in get('gt_metadata.json')['files']}
    assert len(inputs) == len(gts) == len(pairs['rows']) == 150 and set(inputs) == set(gts)
    assert pairs['rows'] == [dict(name=name, input=inputs[name], gt=gts[name]) for name in sorted(inputs)]
    assert choice['name'] == min(inputs) == low.name and choice['file_id'] == inputs[low.name]['id']
    assert choice['declared_utc'] < acquired['acquired_utc'] and choice['pixels_read'] == 0
    assert sha(here / 'pairs_manifest.json') == choice['pairs_manifest_sha256']
    assert sha(low) == acquired['sha256'] and low.stat().st_size == int(inputs[low.name]['size'])
    with Image.open(low) as image:
        assert (image.width, image.height, image.mode) == (3840, 2160, 'RGB')

    records = []
    for method in METHODS:
        receipt_path = out / method / 'receipt.json'
        receipt = json.loads(receipt_path.read_bytes())
        assert receipt['status'] == 'PASS' and receipt['attempts'] == 1
        assert receipt['reference_reads'] == receipt['model_fits'] == 0
        assert receipt['opened'] == [str(low)] and receipt['input_sha256'] == acquired['sha256']
        assert receipt['post_binding_unchanged']
        assert receipt['seconds'] >= 0 and receipt['peak_gpu_allocated_bytes'] > 0 and receipt['peak_gpu_reserved_bytes'] > 0
        for path, digest in receipt['source_binding'].items():
            assert sha(path) == digest, path
        output = out / method / receipt['output']
        assert sha(output) == receipt['output_file_sha256']
        if method == 'ours':
            value = torch.load(output, weights_only=True, map_location='cpu')
            decision = json.loads((out / method / 'decision.json').read_bytes())
            old_value = torch.load(OLD / 'output.pt', weights_only=True, map_location='cpu')
            old_decision = json.loads((OLD / 'decision.json').read_bytes())
            assert tuple(value.shape) == (1, 3, 2160, 3840) and torch.isfinite(value).all()
            assert thash(value) == receipt['output_hash'] == thash(old_value) == old_decision['output_hash']
            assert decision['state_hash'] == old_decision['state_hash']
            assert decision['output_hash'] == old_decision['output_hash']
            assert sha(out / method / 'decision.json') == receipt['decision_sha256']
        else:
            value = np.load(output)
            assert value.shape == (2160, 3840, 3) and value.dtype == np.float32 and np.isfinite(value).all()
            assert hashlib.sha256(value.tobytes()).hexdigest() == receipt['output_hash']
        records.append(dict(method=method, receipt_sha256=sha(receipt_path), output_sha256=receipt['output_hash'], seconds=receipt['seconds'], peak_gpu_allocated_bytes=receipt['peak_gpu_allocated_bytes'], peak_gpu_reserved_bytes=receipt['peak_gpu_reserved_bytes']))

    result = dict(status='PASS', classification='UHDLL_NATIVE_PREFLIGHT_PASS', pairs=150, smoke=low.name, reference_reads=0, model_fits=0, metrics=0, source_binding_checks=True, output_geometry='3840x2160 RGB', ours_repeat_matches_t072b=True, methods=records, verified_utc=utc())
    write(out / 'verification.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--low', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    main(args.low.resolve(), args.out.resolve())
