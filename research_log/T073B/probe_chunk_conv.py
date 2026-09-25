"""Synthetic-only equivalence and native-size PromptIR convolution probe."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

from chunk_conv_schedule import schedule_depthwise
from run_low_only import load_promptir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root))
    torch.manual_seed(23)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = load_promptir(args.checkpoint).cuda().eval()
    small = torch.rand(1, 3, 352, 352, device="cuda")
    with torch.no_grad():
        base = model(small)
    modules = schedule_depthwise(model, rows=256, threshold=0)
    with torch.no_grad():
        equivalent = model(small)
    difference = (base - equivalent).abs().max().item()
    assert torch.equal(base, equivalent), difference
    del small, base, equivalent
    torch.cuda.empty_cache()
    started = time.perf_counter()
    large = torch.rand(1, 3, 2160, 3840, device="cuda")
    with torch.no_grad():
        output = model(large)
    torch.cuda.synchronize()
    assert output.shape == large.shape and torch.isfinite(output).all()
    receipt = {
        "input": "torch.manual_seed(23) synthetic uniform float32 low; no target file",
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "small_shape": [1, 3, 352, 352],
        "small_bitwise_equal": True,
        "small_max_abs_diff": difference,
        "scheduled_modules": modules,
        "native_shape": list(output.shape),
        "native_finite": True,
        "native_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_reserved(),
        "reference_reads": 0,
        "target_model_runs": 0,
        "metrics": 0,
    }
    args.out.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
