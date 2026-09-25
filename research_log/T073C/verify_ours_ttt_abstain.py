"""Independently verify complete-case Ours-TTT outputs and abstention rule."""

import argparse
import gzip
import json
from pathlib import Path

import torch

from research_log.T063A.common import sha, thash


def verify_row(row, expected, step0, output_root):
    assert row["low_name"] == expected["name"] == step0["low_name"]
    assert row["low_sha256"] == expected["sha256"] == step0["low_sha256"]
    directory = output_root / Path(expected["name"]).stem
    assert json.loads((directory / "decision.json").read_bytes()) == row
    output = directory / "output.pt.gz"
    assert sha(output) == row["output_file_sha256"]
    with gzip.open(output, "rb") as stream:
        tensor = torch.load(stream, map_location="cpu", weights_only=True)
    assert list(tensor.shape) == row["shape"] == [1, 3, 2160, 3840]
    assert str(tensor.dtype) == row["dtype"] == "torch.float32"
    assert torch.isfinite(tensor).all()
    assert thash(tensor) == row["output_tensor_sha256"]
    inactive = not any(step0["active"])
    if inactive:
        assert row["decision_status"] == row["decision"]["status"] == "TTT_ABSTAIN_NO_ACTIVE_GATE"
        assert row["decision"]["selected_step"] == 0 and row["optimizer_updates"] == 0
        assert row["active"] == step0["active"]
        assert row["output_tensor_sha256"] == step0["output_tensor_sha256"]
        assert row["output_file_sha256"] == step0["output_file_sha256"]
        assert row["selected_state_sha256"] == thash(torch.zeros(1, 3, 2, 2))
    else:
        assert row["decision_status"] == "TTT_EXECUTED"
        assert 0 <= row["decision"]["selected_step"] <= 27
        assert len(row["selected_state_sha256"]) == 64
    return inactive, output.stat().st_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--low-receipt", type=Path, required=True)
    parser.add_argument("--step0-manifest", type=Path, required=True)
    parser.add_argument("--execution-manifest", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    expected = json.loads(args.low_receipt.read_bytes())["files"]
    step0 = json.loads(args.step0_manifest.read_bytes())
    manifest_path = args.outputs / "output_manifest.json"
    produced = json.loads(manifest_path.read_bytes())
    assert len(expected) == step0["count"] == produced["count"] == 150
    assert len(step0["rows"]) == len(produced["rows"]) == 150
    assert produced["low_receipt_sha256"] == step0["low_receipt_sha256"] == sha(args.low_receipt)
    assert produced["step0_output_manifest_sha256"] == sha(args.step0_manifest)
    assert produced["execution_manifest_sha256"] == step0["execution_manifest_sha256"] == sha(args.execution_manifest)
    assert produced["reference_reads"] == produced["metrics"] == 0
    inactive_count = 0
    total_bytes = 0
    for index, (item, frozen, row) in enumerate(zip(expected, step0["rows"], produced["rows"])):
        assert row["execution_manifest_sha256"] == produced["execution_manifest_sha256"]
        inactive, size = verify_row(row, item, frozen, args.outputs)
        inactive_count += inactive
        total_bytes += size
    assert inactive_count == 49
    assert produced["rows"][0]["output_tensor_sha256"] == "6932c94af73d1f4e8b6d4b4e37eee6521fef82a921e03b9cc202146dbadac580"
    receipt = {
        "classification": "Ours-TTT-complete-case-output-verification-passed",
        "count": 150,
        "ttt_abstain_no_active_gate_count": inactive_count,
        "ttt_executed_count": 150 - inactive_count,
        "output_manifest_sha256": sha(manifest_path),
        "compressed_output_bytes": total_bytes,
        "reference_reads": 0,
        "metrics": 0,
    }
    args.out.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
