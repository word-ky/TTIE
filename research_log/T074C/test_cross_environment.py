"""Observed G4 hash-drift mode: strict default retained, exploratory drift disclosed."""
import pytest
from ours_tuning import default_hash_mismatches
from reference_gate import GateError


def test_strict_and_exploratory_default_hashes():
    items = [{"name": "same.png"}, {"name": "drift.png"}]
    rows = [{"output_tensor_sha256": "a"}, {"output_tensor_sha256": "b"}]
    assert default_hash_mismatches(items, rows, ["a", "b"]) == []
    with pytest.raises(GateError, match="drift.png"):
        default_hash_mismatches(items, rows, ["a", "c"])
    assert default_hash_mismatches(items, rows, ["a", "c"], exploratory=True) == ["drift.png"]
