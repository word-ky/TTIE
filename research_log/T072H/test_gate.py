import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from verify_gate import verify


def test_clean_gpu_gate_is_blocked_without_running_models():
    result = verify()
    assert result["classification"] == "BLOCKED"
    assert result["free_mib"] == [3497, 3499]
    assert result["reference_reads"] == 0
    assert result["metrics"] == 0
