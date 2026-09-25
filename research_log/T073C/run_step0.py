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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--low", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    model = FinalOurs(args.manifest)
    restored, gate = render_step0(model, native_rgb(args.low))
    assert torch.isfinite(restored).all()
    args.out.mkdir(parents=True, exist_ok=False)
    with gzip.open(args.out / "output.pt.gz", "wb", compresslevel=1) as stream:
        torch.save(restored, stream)
    torch.cuda.synchronize()
    receipt = {
        "method": "Ours-Step0",
        "low_name": args.low.name,
        "low_sha256": sha(args.low),
        "execution_manifest_sha256": model.manifest_sha256,
        "state": "frozen CommonRegion2 raw=zeros; zero optimizer updates",
        "active": gate.active.cpu().tolist(),
        "shape": list(restored.shape),
        "dtype": str(restored.dtype),
        "output_tensor_sha256": thash(restored),
        "output_file_sha256": sha(args.out / "output.pt.gz"),
        "whole_run_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_reserved(),
    }
    (args.out / "decision.json").write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
