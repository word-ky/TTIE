import json,hashlib
import pytest
from scripts.bind_t033a import verify

def test_freeze_binding_rejects_replaced_outputs_manifest(tmp_path):
    p=tmp_path/'freeze.json';p.write_text(json.dumps({'completed_utc':'2026-09-14T10:00:00Z'}))
    r={'freeze_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'created_utc':'2026-09-14T11:00:00Z'}
    assert verify(tmp_path,r)['completed_utc']=='2026-09-14T10:00:00Z'
    p.write_text('{}')
    with pytest.raises(AssertionError,match='freeze mismatch'):verify(tmp_path,r)
