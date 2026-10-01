"""Synthetic-only QuadPrior execution probe (T073-E): peak VRAM and repeat determinism.

Creates deterministic synthetic low images at the requested geometries and a
receipt, runs `run_quadprior_batch.py --synthetic` twice in fresh processes
(each with its own task-owned cwd), and compares every output tensor hash.
Never touches a real dataset image.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def synthetic_image(height, width, index):
    y, x = np.indices((height, width), dtype=np.float64)
    base = 12.0 + 10.0 * np.sin(x / 97.0 + index) * np.cos(y / 61.0) + 8.0 * (x / width)
    noise = np.random.default_rng(2000 + index).normal(0.0, 3.0, (height, width, 3))
    bgr = np.stack((base * 0.8, base, base * 1.1), -1) + noise
    return np.clip(bgr * (1 + 2 * (index % 2)), 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--work", type=Path, required=True, help="new directory for the probe")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--coco-checkpoint", type=Path, required=True)
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument("--control-checkpoint", type=Path, required=True)
    parser.add_argument("--sizes", nargs="+", default=["2160x3840", "1080x1916"])
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=False)
    low = args.work / "low"
    low.mkdir()
    files = []
    for index, size in enumerate(args.sizes):
        height, width = (int(v) for v in size.split("x"))
        name = f"synthetic_{index:02d}_{height}x{width}.png"
        assert cv2.imwrite(str(low / name), synthetic_image(height, width, index))
        files.append({"name": name, "sha256": hashlib.sha256((low / name).read_bytes()).hexdigest()})
    receipt = args.work / "synthetic_receipt.json"
    receipt.write_text(json.dumps({"count": len(files), "files": files}, indent=2))

    runs = []
    for repeat in range(args.repeats):
        command = [sys.executable, "-B", str(HERE / "run_quadprior_batch.py"),
                   "--source-root", str(args.source_root), "--coco-checkpoint", str(args.coco_checkpoint),
                   "--vae-checkpoint", str(args.vae_checkpoint),
                   "--control-checkpoint", str(args.control_checkpoint), "--low-dir", str(low),
                   "--low-receipt", str(receipt), "--out", str(args.work / f"run_{repeat}"),
                   "--workdir", str(args.work / f"cwd_{repeat}"), "--synthetic"]
        with (args.work / f"run_{repeat}.log").open("w") as log:
            code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
        manifest_path = args.work / f"run_{repeat}" / "output_manifest.json"
        runs.append({"exit_code": code,
                     "manifest": json.loads(manifest_path.read_bytes()) if manifest_path.exists() else None})
        if code != 0:
            break

    summary = {"sizes": args.sizes, "exit_codes": [r["exit_code"] for r in runs], "images": []}
    if all(r["exit_code"] == 0 for r in runs):
        for position, item in enumerate(files):
            rows = [r["manifest"]["rows"][position] for r in runs]
            summary["images"].append({
                "name": item["name"],
                "native_shape": rows[0]["native_shape"],
                "peak_reserved_bytes": [row["peak_gpu_memory_bytes"] for row in rows],
                "process_seconds": [row["process_seconds"] for row in rows],
                "prequant_dtype": rows[0]["prequant_dtype"],
                "bitwise_repeatable": len({row["native_tensor_sha256"] for row in rows}) == 1,
            })
        summary["model_state_sha256"] = runs[0]["manifest"]["model_state_sha256"]
        summary["environment"] = runs[0]["manifest"]["environment"]
    (args.work / "probe_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    sys.exit(0 if all(r["exit_code"] == 0 for r in runs) else 1)


if __name__ == "__main__":
    main()
