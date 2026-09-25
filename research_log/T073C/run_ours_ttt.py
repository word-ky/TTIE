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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--low", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    model = FinalOurs(args.manifest)
    result = model(native_rgb(args.low))
    selected = result["image"].clone().contiguous()
    assert torch.isfinite(selected).all()
    args.out.mkdir(parents=True, exist_ok=False)
    with gzip.open(args.out / "output.pt.gz", "wb", compresslevel=1) as stream:
        torch.save(selected, stream)
    torch.cuda.synchronize()
    receipt = {
        "method": "Ours-TTT",
        "low_name": args.low.name,
        "low_sha256": sha(args.low),
        "execution_manifest_sha256": model.manifest_sha256,
        "decision": result["decision"],
        "selected_state_sha256": thash(result["state"]),
        "shape": list(selected.shape),
        "dtype": str(selected.dtype),
        "output_tensor_sha256": thash(selected),
        "output_file_sha256": sha(args.out / "output.pt.gz"),
        "whole_run_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_reserved(),
    }
    (args.out / "decision.json").write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
