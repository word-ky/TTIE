"""Synthetic-only equivalence and native-size PromptIR convolution probe."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

from chunk_conv_schedule import schedule_depthwise, schedule_feedforward
from run_low_only import load_promptir
from spill_forward_schedule import schedule_skip_spill


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--spill", action="store_true")
    parser.add_argument("--chunk-ffn", action="store_true")
    parser.add_argument("--small-size", type=int, default=352)
    parser.add_argument("--pattern", choices=("random", "gradient", "dark"), default="random")
    parser.add_argument("--skip-native", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root))
    torch.manual_seed(23)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = load_promptir(args.checkpoint).cuda().eval()
    frozen_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if args.pattern == "random":
        small = torch.rand(1, 3, args.small_size, args.small_size, device="cuda")
    elif args.pattern == "dark":
        small = 0.1 * torch.rand(1, 3, args.small_size, args.small_size, device="cuda")
    else:
        line = torch.linspace(0, 1, args.small_size, device="cuda")
        small = torch.stack((line[None, :].expand(args.small_size, -1), line[:, None].expand(-1, args.small_size), (line[None, :] + line[:, None]) / 2), dim=0)[None]
    with torch.no_grad():
        base = model(small)
    modules = schedule_depthwise(model, rows=256, threshold=0)
    ffn_modules = schedule_feedforward(model, rows=64, threshold=0) if args.chunk_ffn else []
    if args.spill:
        schedule_skip_spill(model, threshold=0)
    with torch.no_grad():
        equivalent = model(small)
    error = (base - equivalent).abs()
    difference = error.max().item()
    mean_difference = error.mean().item()
    quantized_mismatch = (
        torch.clamp(base * 255, 0, 255).to(torch.uint8)
        != torch.clamp(equivalent * 255, 0, 255).to(torch.uint8)
    ).float().mean().item()
    exact = torch.equal(base, equivalent)
    if not args.chunk_ffn:
        assert exact, difference
    else:
        assert difference < 0.001, difference
    del small, base, equivalent
    torch.cuda.empty_cache()
    receipt = {
        "input": "torch.manual_seed(23) synthetic uniform float32 low; no target file",
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "small_shape": [1, 3, args.small_size, args.small_size],
        "small_pattern": args.pattern,
        "small_bitwise_equal": exact,
        "small_max_abs_diff": difference,
        "small_mean_abs_diff": mean_difference,
        "small_uint8_mismatch_fraction": quantized_mismatch,
        "scheduled_modules": modules,
        "scheduled_ffn_modules": ffn_modules,
        "skip_spill": args.spill,
        "reference_reads": 0,
        "target_model_runs": 0,
        "metrics": 0,
    }
    if not args.skip_native:
        started = time.perf_counter()
        large = torch.rand(1, 3, 2160, 3840, device="cuda")
        with torch.no_grad():
            output = model(large)
        torch.cuda.synchronize()
        assert output.shape == large.shape and torch.isfinite(output).all()
        receipt["native_shape"] = list(output.shape)
        receipt["native_finite"] = True
        receipt["native_output_tensor_sha256"] = hashlib.sha256(output.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        receipt["native_seconds"] = time.perf_counter() - started
    receipt["peak_gpu_memory_bytes"] = torch.cuda.max_memory_reserved()
    receipt["model_state_unchanged"] = all(
        torch.equal(value.detach().cpu(), frozen_state[key])
        for key, value in model.state_dict().items()
    )
    assert receipt["model_state_unchanged"]
    args.out.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
