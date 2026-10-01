"""T074-B generic static PromptIR low-only outputs (derived from T073B; see make_generic.py)."""

import os as _os
import sys as _sys

# T074-B: frozen UHD-LL helper modules (T073B/shared on the GPU host).
_sys.path.insert(0, _os.environ.get("T073B_SHARED", str(__import__("pathlib").Path(__file__).resolve().parent.parent / "T073B")))

import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

from chunk_conv_schedule import schedule_depthwise, schedule_feedforward
from low_only_loader import LowOnlyPromptTestDataset
from run_low_only import load_promptir
from spill_forward_schedule import schedule_skip_spill


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--low-dir", type=Path, required=True)
    parser.add_argument("--low-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke-one", action="store_true")
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--memory-schedule", action="store_true",
                        help="UHD-LL 4K schedule; default is the official unscheduled forward")
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root))
    torch.manual_seed(23)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    expected = json.loads(args.low_receipt.read_bytes())["files"]
    assert len(expected) == args.expected_count
    assert all(i["height"] % 16 == 0 and i["width"] % 16 == 0 for i in expected), "official PromptIR test loader would center-crop to multiples of 16"
    if args.smoke_one:
        expected = expected[:1]
    dataset = LowOnlyPromptTestDataset(args.low_dir)
    assert dataset.names[:len(expected)] == [item["name"] for item in expected]
    model = load_promptir(args.checkpoint).cuda().eval()
    frozen_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if args.memory_schedule:
        schedule_depthwise(model, rows=256, threshold=250_000_000)
        schedule_feedforward(model, rows=64, threshold=250_000_000)
        schedule_skip_spill(model, threshold=1_000_000)
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, item in enumerate(expected):
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        name, low = dataset[index]
        low_path = args.low_dir / name
        low_sha = sha256_file(low_path)
        assert name == item["name"] and low_sha == item["sha256"]
        assert low.shape == (3, item["height"], item["width"])
        with torch.no_grad():
            output = model(low.unsqueeze(0).cuda())
        torch.cuda.synchronize()
        assert output.shape == (1, 3, item["height"], item["width"])
        assert output.dtype == torch.float32 and torch.isfinite(output).all()
        output_cpu = output.detach().cpu().contiguous()
        del output
        tensor_sha = hashlib.sha256(output_cpu.numpy().tobytes()).hexdigest()
        output_dir = args.out / Path(name).stem
        output_dir.mkdir()
        output_file = output_dir / "output.pt.gz"
        with gzip.open(output_file, "wb", compresslevel=1) as stream:
            torch.save(output_cpu, stream)
        del output_cpu, low
        row = {
            "low_name": name,
            "low_sha256": low_sha,
            "shape": [1, 3, item["height"], item["width"]],
            "dtype": "torch.float32",
            "output_tensor_sha256": tensor_sha,
            "output_file_sha256": sha256_file(output_file),
            "whole_run_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_reserved(),
        }
        (output_dir / "decision.json").write_text(json.dumps(row, indent=2))
        rows.append(row)
        print(f"{index + 1}/{len(expected)} {name} {row['whole_run_seconds']:.3f}s", flush=True)
    state_unchanged = all(
        torch.equal(value.detach().cpu(), frozen_state[key])
        for key, value in model.state_dict().items()
    )
    assert state_unchanged
    manifest = {
        "method": "PromptIR-static-five-task-generic",
        "memory_schedule": bool(args.memory_schedule),
        "count": len(rows),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "low_receipt_sha256": sha256_file(args.low_receipt),
        "model_state_unchanged": state_unchanged,
        "state_key_count": len(frozen_state),
        "reference_reads": 0,
        "metrics": 0,
        "rows": rows,
    }
    manifest_path = args.out / "output_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"OUTPUT_MANIFEST_SHA256 {sha256_file(manifest_path)}", flush=True)


if __name__ == "__main__":
    main()
