"""Verify compact TTT output against unmodified frozen entrypoint evidence."""

import argparse
import gzip
import json
from pathlib import Path

import torch

from research_log.T063A.common import thash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-decision", type=Path, required=True)
    parser.add_argument("--compact-decision", type=Path, required=True)
    parser.add_argument("--compact-output", type=Path, required=True)
    args = parser.parse_args()
    original = json.loads(args.original_decision.read_bytes())
    compact = json.loads(args.compact_decision.read_bytes())
    assert compact["decision"] == {key: original[key] for key in compact["decision"]}
    assert compact["selected_state_sha256"] == original["state_hash"]
    assert compact["output_tensor_sha256"] == original["output_hash"]
    with gzip.open(args.compact_output, "rb") as stream:
        image = torch.load(stream, map_location="cpu", weights_only=True)
    assert thash(image) == original["output_hash"]
    assert torch.isfinite(image).all()
    print("TTT_SELECTED_STATE_OUTPUT_AND_GZIP_ROUNDTRIP_PASS")


if __name__ == "__main__":
    main()
