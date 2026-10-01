"""Complete-case Ours-TTT with the prospectively sealed no-active rule."""

import argparse
import gzip
import json
import shutil
import time
from pathlib import Path

import torch


def run_case(model, low_path, expected, step0, step0_root, output_dir, run_active):
    from research_log.T063A.common import sha, thash

    started = time.perf_counter()
    assert low_path.name == expected["name"] == step0["low_name"]
    low_hash = sha(low_path)
    assert low_hash == expected["sha256"] == step0["low_sha256"]
    assert step0["execution_manifest_sha256"] == model.manifest_sha256
    if any(step0["active"]):
        receipt = run_active(model, low_path, output_dir, expected_low_sha256=low_hash)
        receipt["decision_status"] = "TTT_EXECUTED"
        (output_dir / "decision.json").write_text(json.dumps(receipt, indent=2))
        return receipt

    source = step0_root / Path(low_path.name).stem / "output.pt.gz"
    assert sha(source) == step0["output_file_sha256"]
    with gzip.open(source, "rb") as stream:
        step0_tensor = torch.load(stream, map_location="cpu", weights_only=True)
    assert thash(step0_tensor) == step0["output_tensor_sha256"]
    assert list(step0_tensor.shape) == [1, 3, 2160, 3840]
    assert step0_tensor.dtype == torch.float32 and torch.isfinite(step0_tensor).all()
    output_dir.mkdir(parents=True, exist_ok=False)
    destination = output_dir / "output.pt.gz"
    shutil.copyfile(source, destination)
    assert sha(destination) == step0["output_file_sha256"]
    zero_state_sha = thash(torch.zeros(1, 3, 2, 2))
    receipt = {
        "method": "Ours-TTT",
        "low_name": low_path.name,
        "low_sha256": low_hash,
        "execution_manifest_sha256": model.manifest_sha256,
        "decision_status": "TTT_ABSTAIN_NO_ACTIVE_GATE",
        "decision": {"selected_step": 0, "status": "TTT_ABSTAIN_NO_ACTIVE_GATE"},
        "active": step0["active"],
        "optimizer_updates": 0,
        "selected_state_sha256": zero_state_sha,
        "shape": step0["shape"],
        "dtype": step0["dtype"],
        "output_tensor_sha256": step0["output_tensor_sha256"],
        "output_file_sha256": step0["output_file_sha256"],
        "step0_output_manifest_sha256": step0["output_manifest_sha256"],
        "whole_run_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_reserved(),
    }
    (output_dir / "decision.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def main():
    from research_log.T063A.common import sha
    from research_log.T070A.infer import FinalOurs
    from research_log.T073C.run_ours_ttt import run_one

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--low-receipt", type=Path, required=True)
    parser.add_argument("--low-dir", type=Path, required=True)
    parser.add_argument("--step0-manifest", type=Path, required=True)
    parser.add_argument("--step0-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke-count", type=int, default=0)
    args = parser.parse_args()
    expected = json.loads(args.low_receipt.read_bytes())["files"]
    frozen_step0 = json.loads(args.step0_manifest.read_bytes())
    assert len(expected) == frozen_step0["count"] == 150
    assert frozen_step0["low_receipt_sha256"] == sha(args.low_receipt)
    step0_manifest_hash = sha(args.step0_manifest)
    assert step0_manifest_hash == "76a1ee7b12f41991499802024811321dcce3b0af0bae2c06ab41dba719180cce"
    model = FinalOurs(args.manifest)
    assert frozen_step0["execution_manifest_sha256"] == model.manifest_sha256
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    cohort = expected[:args.smoke_count] if args.smoke_count else expected
    for index, item in enumerate(cohort):
        torch.cuda.reset_peak_memory_stats()
        step0 = dict(frozen_step0["rows"][index], output_manifest_sha256=step0_manifest_hash)
        row = run_case(
            model,
            args.low_dir / item["name"],
            item,
            step0,
            args.step0_root,
            args.out / Path(item["name"]).stem,
            run_one,
        )
        rows.append(row)
        print(f"{index + 1}/{len(cohort)} {item['name']} {row['decision_status']}", flush=True)
    summary = {
        "method": "Ours-TTT-with-no-active-abstention",
        "count": len(rows),
        "execution_manifest_sha256": model.manifest_sha256,
        "low_receipt_sha256": sha(args.low_receipt),
        "step0_output_manifest_sha256": step0_manifest_hash,
        "reference_reads": 0,
        "metrics": 0,
        "rows": rows,
    }
    path = args.out / "output_manifest.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"OUTPUT_MANIFEST_SHA256 {sha(path)}", flush=True)


if __name__ == "__main__":
    main()
