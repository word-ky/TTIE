"""Separate low-only extraction; no reference manifest or normal entry is read."""
import json,hashlib,zipfile
from pathlib import Path


def main():
    rows=json.loads(Path('research_log/T063D/low_inputs.json').read_bytes())
    root=Path('/media/wenchang/F/wjq/TTIE/shared/t063d/low')
    with zipfile.ZipFile('/media/wenchang/F/wjq/TTIE/shared/t022a/LOL-v2.zip') as z:
        for row in rows:
            data=z.read('LOL-v2/Real_captured/'+row['low'])
            assert hashlib.sha256(data).hexdigest()==row['low_sha256']
            p=root/row['low'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    print('100 low entries extracted; normal entries0',flush=True)


if __name__=='__main__':main()
