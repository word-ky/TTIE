"""Frozen T070-A zero-update forward on one low image, without target references."""

import argparse
import gzip
import json
import time
from pathlib import Path

import torch

from research_log.T063A.common import sha, thash
from research_log.T070A.infer import FinalOurs
from ttie.common_gain import CommonRegion2
from ttie.lolv2_gamma_core import native_rgb
from ttie.semantic_ttt import FixedObjective


def render_step0(model, low):
    image = low.cuda()
    gate = FixedObjective(model.scorer, image, model.gate)
    with torch.no_grad():
        restored = CommonRegion2(gate.active).to(image)(image).detach().cpu()
    return restored, gate


def run_one(model, low_path, output_dir, expected_low_sha256=None):
    started = time.perf_counter()
    low_sha256 = sha(low_path)
    if expected_low_sha256 is not None:
        assert low_sha256 == expected_low_sha256
    restored, gate = render_step0(model, native_rgb(low_path))
    assert torch.isfinite(restored).all()
    output_dir.mkdir(parents=True, exist_ok=False)
    with gzip.open(output_dir / "output.pt.gz", "wb", compresslevel=1) as stream:
        torch.save(restored, stream)
    torch.cuda.synchronize()
    receipt = {
        "method": "Ours-Step0",
        "low_name": low_path.name,
        "low_sha256": low_sha256,
        "execution_manifest_sha256": model.manifest_sha256,
        "state": "frozen CommonRegion2 raw=zeros; zero optimizer updates",
        "active": gate.active.cpu().tolist(),
        "shape": list(restored.shape),
        "dtype": str(restored.dtype),
        "output_tensor_sha256": thash(restored),
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
