import json
from pathlib import Path

def test_canonical_complete_pairing_and_predeclared_selection():
    d=Path(__file__).parent
    p=json.loads((d/'pairs_manifest.json').read_bytes());c=json.loads((d/'smoke_selection.json').read_bytes());r=json.loads((d/'input_receipt.json').read_bytes())
    assert len(p['rows'])==150 and len({x['name'] for x in p['rows']})==150
    assert all(x['input']['name']==x['gt']['name']==x['name'] for x in p['rows'])
    assert c['name']==min(x['name'] for x in p['rows'])==r['name']
    assert c['declared_utc']<r['acquired_utc'] and c['pixels_read']==0
    assert r['geometry']==dict(width=3840,height=2160,mode='RGB',format='JPEG')
    assert p['reference_reads']==r['reference_reads']==r['reference_downloads']==0


def test_repaired_writer_uses_distinct_initial_and_final_receipts():
    source = (Path(__file__).parent / 'run.py').read_text(encoding='utf-8')
    assert "initial_receipt.json" in source
    assert "write(out/'receipt.json',record)" in source
