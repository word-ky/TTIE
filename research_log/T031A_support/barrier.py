"""Bind the support freeze to the separately saved deployment receipt."""
import hashlib
import json


def verify_freeze(support, deployment):
    expected = json.loads(deployment.read_bytes())['support_freeze_sha256']
    actual = hashlib.sha256((support / 'freeze.json').read_bytes()).hexdigest()
    assert actual == expected, 'support freeze differs from deployment receipt'
    return actual
