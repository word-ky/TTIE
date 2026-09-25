"""One execution-only native PromptIR smoke on the first target low image."""

import argparse
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
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root))
    torch.manual_seed(23)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dataset = LowOnlyPromptTestDataset(args.low_dir)
    name, low = dataset[0]
    low_path = args.low_dir / name
    assert low.shape == (3, 2160, 3840), low.shape
    model = load_promptir(args.checkpoint).cuda().eval()
    frozen_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    conv_modules = schedule_depthwise(model, rows=256, threshold=250_000_000)
    ffn_modules = schedule_feedforward(model, rows=64, threshold=250_000_000)
    schedule_skip_spill(model, threshold=1_000_000)
    low = low.unsqueeze(0).cuda()
    start = time.perf_counter()
    with torch.no_grad():
        restored = model(low)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    assert restored.shape == low.shape and restored.dtype == torch.float32
    assert torch.isfinite(restored).all()
    state_unchanged = all(
        torch.equal(value.detach().cpu(), frozen_state[key])
        for key, value in model.state_dict().items()
    )
    assert state_unchanged
    receipt = {
        "mode": "static_promptir_native_schedule_execution_only",
        "low_name": name,
        "low_file_sha256": sha256_file(low_path),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "input_shape": list(low.shape),
        "output_shape": list(restored.shape),
        "output_dtype": str(restored.dtype),
        "output_finite": True,
        "output_tensor_sha256": hashlib.sha256(restored.cpu().contiguous().numpy().tobytes()).hexdigest(),
        "forward_seconds": seconds,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_reserved(),
        "model_state_unchanged": state_unchanged,
        "state_key_count": len(frozen_state),
        "scheduled_conv_count": len(conv_modules),
        "scheduled_ffn_count": len(ffn_modules),
        "reference_reads": 0,
        "metrics": 0,
    }
    args.out.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
