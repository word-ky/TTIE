"""Run unchanged frozen T070-A inference, saving only the selected tensor losslessly."""

import argparse
import gzip
import json
import time
from pathlib import Path

import torch

from research_log.T063A.common import sha, thash
from research_log.T070A.infer import FinalOurs
from ttie.lolv2_gamma_core import native_rgb


def run_one(model, low_path, output_dir, expected_low_sha256=None):
    started = time.perf_counter()
    low_sha256 = sha(low_path)
    if expected_low_sha256 is not None:
        assert low_sha256 == expected_low_sha256
    result = model(native_rgb(low_path))
    selected = result["image"].clone().contiguous()
    assert torch.isfinite(selected).all()
    output_dir.mkdir(parents=True, exist_ok=False)
    with gzip.open(output_dir / "output.pt.gz", "wb", compresslevel=1) as stream:
        torch.save(selected, stream)
    torch.cuda.synchronize()
    receipt = {
        "method": "Ours-TTT",
        "low_name": low_path.name,
        "low_sha256": low_sha256,
        "execution_manifest_sha256": model.manifest_sha256,
        "decision": result["decision"],
        "selected_state_sha256": thash(result["state"]),
        "shape": list(selected.shape),
        "dtype": str(selected.dtype),
        "output_tensor_sha256": thash(selected),
        "output_file_sha256": sha(output_dir / "output.pt.gz"),
        "whole_run_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_reserved(),
    }
    (output_dir / "decision.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--low", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    model = FinalOurs(args.manifest)
    run_one(model, args.low, args.out)


if __name__ == "__main__":
    main()
