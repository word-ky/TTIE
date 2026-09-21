import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from verify_manifest import verify


def test_smid_protocol_block_is_reproducible():
    receipt = verify()
    assert receipt["classification"] == "BLOCKED"
    assert receipt["scene_count"] == 49
    assert receipt["reference_reads"] == 0
    assert receipt["metrics"] == 0


def test_manifest_is_low_only_and_does_not_claim_payload_coverage():
    manifest = json.loads((Path(__file__).parent / "manifest.json").read_text(encoding="utf-8"))
    candidate = manifest["candidate_low_only_manifest"]
    assert candidate["low_payload_enumerated"] is False
    assert candidate["low_sha256"] is None
    assert candidate["reference_payload_opened"] is False
