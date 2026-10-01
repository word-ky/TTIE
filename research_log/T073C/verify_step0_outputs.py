"""Independently verify frozen low-only UHD-LL Step0 tensor outputs."""

import argparse
import gzip
import json
from pathlib import Path

import torch

from research_log.T063A.common import sha, thash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--low-receipt", type=Path, required=True)
    parser.add_argument("--execution-manifest", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    expected = json.loads(args.low_receipt.read_bytes())["files"]
    manifest_path = args.outputs / "output_manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    names = [item["name"] for item in expected]
    assert len(names) == len(set(names)) == 150
    assert manifest["count"] == len(manifest["rows"]) == 150
    assert manifest["execution_manifest_sha256"] == sha(args.execution_manifest)
    assert manifest["low_receipt_sha256"] == sha(args.low_receipt)
    total_bytes = 0
    for item, row in zip(expected, manifest["rows"]):
        assert row["low_name"] == item["name"]
        assert row["low_sha256"] == item["sha256"]
        assert row["execution_manifest_sha256"] == manifest["execution_manifest_sha256"]
        assert row["shape"] == [1, 3, 2160, 3840]
        assert row["dtype"] == "torch.float32"
        directory = args.outputs / Path(item["name"]).stem
        assert json.loads((directory / "decision.json").read_bytes()) == row
        output = directory / "output.pt.gz"
        assert sha(output) == row["output_file_sha256"]
        total_bytes += output.stat().st_size
        with gzip.open(output, "rb") as stream:
            tensor = torch.load(stream, map_location="cpu", weights_only=True)
        assert list(tensor.shape) == row["shape"] and tensor.dtype == torch.float32
        assert torch.isfinite(tensor).all()
        assert thash(tensor) == row["output_tensor_sha256"]
    receipt = {
        "classification": "UHDLL_OURS_STEP0_OUTPUTS_VERIFIED",
        "count": len(expected),
        "output_manifest_sha256": sha(manifest_path),
        "execution_manifest_sha256": sha(args.execution_manifest),
        "low_receipt_sha256": sha(args.low_receipt),
        "compressed_output_bytes": total_bytes,
        "reference_payload_read": False,
        "metrics_computed": False,
    }
    args.out.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
