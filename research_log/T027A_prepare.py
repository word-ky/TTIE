"""Select names without decoding; extract exactly eight non-validation lows."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

archive, split_path, out = map(Path, sys.argv[1:])
split = json.loads(split_path.read_text())
validation = {x['low'] for x in split['selected']}
prefix = 'LOL-v2/Real_captured/'
with zipfile.ZipFile(archive) as z:
    train = [n[len(prefix):] for n in z.namelist()
             if n.startswith(prefix + 'Train/Low/') and n.endswith('.png')]
    pool = sorted(set(train) - validation)
    assert len(train) == 689 and len(pool) == 589
    selected = sorted(pool, key=lambda p: hashlib.sha256(
        ('TTIE-T027A-smoke|' + p).encode()).hexdigest())[:8]
    rows = []
    for name in selected:
        data = z.read(prefix + name)
        p = out / 'low' / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        rows.append({'low': name, 'sha256': hashlib.sha256(data).hexdigest(),
                     'selection_hash': hashlib.sha256(('TTIE-T027A-smoke|' + name).encode()).hexdigest()})
receipt = {'task': 'T027-A', 'selection_prefix': 'TTIE-T027A-smoke|',
           'pool_size': len(pool), 'train_size': len(train), 'validation_size': len(validation),
           'validation_overlap': sorted(set(selected) & validation),
           'test_overlap': [p for p in selected if not p.startswith('Train/Low/')],
           'split_sha256': hashlib.sha256(split_path.read_bytes()).hexdigest(),
           'zip_members_read': [prefix + p for p in selected], 'selected': rows}
(out / 'smoke_manifest.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
