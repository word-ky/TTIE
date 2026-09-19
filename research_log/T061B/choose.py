"""One source-only stopping-step freeze. Standard-library bootstrap precedes guard."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

ACCEPTED = '23c98d16c159e8f51c5147647281ae749741dade'
RULE_COMMIT = '5da5a57e0b481a98d5087aacf7afde8771f80217'
RULE = 'For k=0..40 compute unweighted mean source PSNR across all 60 literal method-A/T036 anchors; choose argmax with earliest-step tie breaking.'
INPUTS = {
    'research_log/T060DR2/evidence/metrics.json': {
        'git_blob': '8db7564a40555aafe52805b58ea3974dcfe84ee1',
        'sha256': '7ae8d7d7574382067e8eeff9dc2ef818bf8689a410c9895c2f1130d7f3e91d0c'},
    'research_log/T060DR2/evidence/freeze.json': {
        'git_blob': 'e1ee98eed078eabdaf2c0bcbdd27e7c0128dbf17',
        'sha256': '9570a0ffc1f5e2fd342458d5a35dffd82c24f59fb38f1fa15566af08665abeb5'},
}

def digest(data):
    return hashlib.sha256(data).hexdigest()

def utc():
    return datetime.now(timezone.utc).isoformat()

def install_guard(allowed, output):
    allowed = {Path(p).resolve() for p in allowed}
    output = Path(output).resolve()
    reads = []
    def guard(event, args):
        if event != 'open':
            return
        name, mode, flags = args
        if isinstance(name, int):
            raise PermissionError('File-descriptor open is not allow-listed')
        path = Path(os.fsdecode(name)).resolve()
        writing = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        in_output = path == output or output in path.parents
        if (writing and not in_output) or (not writing and path not in allowed and not in_output):
            raise PermissionError('Non-allow-listed file access: ' + str(path))
        if not writing:
            reads.append(str(path))
    sys.addaudithook(guard)
    return reads

def persist(path, data):
    with path.open('xb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())

def select(rows, frozen):
    assert len(rows) == len(frozen) == 60
    assert [r['order'] for r in rows] == list(range(60))
    assert len({r['bank_index'] for r in rows}) == 60
    matrix = []
    for r, f in zip(rows, frozen):
        assert all(r[k] == f[k] for k in ['order', 'index', 'bank_index', 'image_id'])
        assert f['methods']['A']['states'] == 41 and f['methods']['A']['updates'] == 40
        values = r['A']['psnr']
        assert len(values) == 41 and all(math.isfinite(v) for v in values)
        matrix.append(values)
    means = [math.fsum(row[k] for row in matrix) / 60 for k in range(41)]
    ties = [k for k, v in enumerate(means) if v == max(means)]
    return means, ties, ties[0]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--script-commit', required=True)
    p.add_argument('--script-sha256', required=True)
    a = p.parse_args()
    root, out = a.root.resolve(), a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    reads = install_guard([root / n for n in INPUTS], out)
    started = utc()
    loaded = {}
    for name, expected in INPUTS.items():
        data = (root / name).read_bytes()
        assert digest(data) == expected['sha256'], name
        assert hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == expected['git_blob']
        loaded[name] = json.loads(data)
    rows = loaded['research_log/T060DR2/evidence/metrics.json']
    freeze = loaded['research_log/T060DR2/evidence/freeze.json']
    means, ties, step = select(rows, freeze['rows'])
    manifest = dict(task='T061-B', status='PASS', accepted_commit=ACCEPTED,
        rule_commit=RULE_COMMIT, rule=RULE, rule_sha256=digest(RULE.encode()),
        script_commit=a.script_commit, script_sha256=a.script_sha256,
        inputs=INPUTS, allow_list=list(INPUTS), output_directory=str(out),
        started_utc=started, frozen_utc=utc(), anchors=60, states_per_anchor=41,
        mean_source_psnr=means, tie_set=ties, k_star=step,
        data_reads=reads.copy(), target_data_reads=0,
        note='Historical T061-A development-summary exposure is disclosed; rule predates exposure. Only method A PSNR is used; accepted metrics container also includes unused method B fields.')
    data = (json.dumps(manifest, indent=2, allow_nan=False) + '\n').encode()
    persist(out / 'manifest.json', data)
    manifest_sha = digest((out / 'manifest.json').read_bytes())
    assert manifest_sha == digest(data)
    persist(out / 'manifest.sha256', (manifest_sha + '\n').encode())
    print(json.dumps(dict(k_star=step, manifest_sha256=manifest_sha, frozen_utc=manifest['frozen_utc'])))

if __name__ == '__main__':
    main()
