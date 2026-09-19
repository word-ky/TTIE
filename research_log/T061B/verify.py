"""Independent Decimal arithmetic, separately read and hash the frozen sources."""
import argparse
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
from choose import INPUTS, RULE, RULE_COMMIT, ACCEPTED, install_guard, persist, utc

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    root, out = a.root.resolve(), a.out.resolve()
    reads = install_guard([root / n for n in INPUTS], out)
    raw = (out / 'manifest.json').read_bytes()
    manifest = json.loads(raw)
    mh = hashlib.sha256(raw).hexdigest()
    assert mh == (out / 'manifest.sha256').read_text().strip()
    assert manifest['inputs'] == INPUTS and manifest['allow_list'] == list(INPUTS)
    assert manifest['rule'] == RULE and manifest['rule_commit'] == RULE_COMMIT
    assert manifest['accepted_commit'] == ACCEPTED
    assert manifest['rule_sha256'] == hashlib.sha256(RULE.encode()).hexdigest()
    objects = []
    for name, expected in INPUTS.items():
        data = (root / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == expected['sha256']
        assert hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == expected['git_blob']
        objects.append(json.loads(data))
    metrics, frozen = objects[0], objects[1]['rows']
    assert len(metrics) == len(frozen) == 60
    assert {r['order'] for r in metrics} == set(range(60))
    assert len({r['bank_index'] for r in metrics}) == 60
    expected_ids = [(r['order'], r['index'], r['bank_index'], r['image_id']) for r in frozen]
    assert [(r['order'], r['index'], r['bank_index'], r['image_id']) for r in metrics] == expected_ids
    columns = [[] for _ in range(41)]
    for row in metrics:
        assert len(row['A']['psnr']) == 41
        for k, v in enumerate(row['A']['psnr']):
            d = Decimal.from_float(v)
            assert d.is_finite()
            columns[k].append(d)
    with localcontext() as context:
        context.prec = 80
        means = [float(sum(c, Decimal(0)) / Decimal(60)) for c in columns]
    errors = [abs(x-y) for x, y in zip(means, manifest['mean_source_psnr'])]
    assert len(manifest['mean_source_psnr']) == 41 and max(errors) <= 1e-12
    ties = [k for k in range(41) if means[k] == max(means)]
    assert ties == manifest['tie_set'] and min(ties) == manifest['k_star']
    receipt = dict(status='PASS', anchors=60, state_values=2460,
        k_star=min(ties), mean_max_abs_error=max(errors), independent_means=means,
        manifest_sha256=mh, completed_utc=utc(), data_reads=reads.copy(), target_data_reads=0,
        method='80-digit Decimal sum of binary64 inputs, divided by 60; independent column assembly and argmax')
    persist(out / 'verification.json', (json.dumps(receipt, indent=2) + '\n').encode())
    print(json.dumps({k:receipt[k] for k in ['status', 'k_star', 'mean_max_abs_error', 'manifest_sha256']}))

if __name__ == '__main__':
    main()
