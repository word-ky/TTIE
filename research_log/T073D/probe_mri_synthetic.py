"""Synthetic-only MR. Illuminate execution probe (T073-D): peak VRAM and repeat determinism.

Creates deterministic synthetic low images at the requested geometries, a
receipt for them, then runs `run_mri_native_batch.py --synthetic` twice in
fresh processes and compares every output tensor hash. Never touches a real
dataset image. `--fake-vae-for-memory-only` writes a randomly initialised
official-architecture VAE checkpoint (seed 0) so peak memory can be measured
before the author checkpoint arrives; such a run says nothing about outputs.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent


def synthetic_image(height, width, index):
    y, x = np.indices((height, width), dtype=np.float64)
    base = 12.0 + 10.0 * np.sin(x / 97.0 + index) * np.cos(y / 61.0) + 8.0 * (x / width)
    noise = np.random.default_rng(1000 + index).normal(0.0, 3.0, (height, width, 3))
    rgb = np.stack((base * 1.1, base, base * 0.8), -1) + noise
    if index % 2:
        rgb = rgb * 3.0  # second image stays above the official mean-30 brightening threshold
    return np.clip(rgb, 0, 255).astype(np.uint8)


def write_fake_vae(source_root, path):
    import torch  # noqa: PLC0415
    sys.path.insert(0, str(source_root))
    from finetuned_vae.autoencoder import AutoencoderKL  # noqa: PLC0415
    torch.manual_seed(0)
    vae = AutoencoderKL(load_checkpoint=False)
    torch.save({"state_dict": {"my_vae." + k: v for k, v in vae.state_dict().items()},
                "note": "RANDOM INIT, MEMORY PROBE ONLY"}, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--work", type=Path, required=True, help="new directory for the probe")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--sd-snapshot", type=Path, required=True)
    parser.add_argument("--vae-checkpoint", type=Path)
    parser.add_argument("--fake-vae-for-memory-only", action="store_true")
    parser.add_argument("--sizes", nargs="+", default=["2160x3840", "2160x3840"],
                        help="HxW per synthetic image, e.g. 2160x3840 2848x4256")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    assert bool(args.vae_checkpoint) != bool(args.fake_vae_for_memory_only), "choose one VAE source"
    args.work.mkdir(parents=True, exist_ok=False)
    low = args.work / "low"
    low.mkdir()
    files = []
    for index, size in enumerate(args.sizes):
        height, width = (int(v) for v in size.split("x"))
        name = f"synthetic_{index:02d}_{height}x{width}.png"
        Image.fromarray(synthetic_image(height, width, index)).save(low / name)
        files.append({"name": name, "sha256": hashlib.sha256((low / name).read_bytes()).hexdigest()})
    receipt = args.work / "synthetic_receipt.json"
    receipt.write_text(json.dumps({"count": len(files), "files": files}, indent=2))
    vae = args.vae_checkpoint
    if args.fake_vae_for_memory_only:
        vae = args.work / "fake_random_vae.ckpt"
        write_fake_vae(args.source_root, vae)

    runs = []
    for repeat in range(args.repeats):
        out = args.work / f"run_{repeat}"
        command = [sys.executable, "-B", str(HERE / "run_mri_native_batch.py"),
                   "--source-root", str(args.source_root), "--vae-checkpoint", str(vae),
                   "--sd-snapshot", str(args.sd_snapshot), "--low-dir", str(low),
                   "--low-receipt", str(receipt), "--out", str(out), "--synthetic"]
        with (args.work / f"run_{repeat}.log").open("w") as log:
            code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
        manifest_path = out / "output_manifest.json"
        manifest = json.loads(manifest_path.read_bytes()) if manifest_path.exists() else None
        runs.append({"exit_code": code, "manifest": manifest})
        if code != 0:
            break

    summary = {"sizes": args.sizes, "fake_vae_memory_only": args.fake_vae_for_memory_only,
               "exit_codes": [r["exit_code"] for r in runs], "images": []}
    if all(r["exit_code"] == 0 for r in runs):
        for position, item in enumerate(files):
            rows = [r["manifest"]["rows"][position] for r in runs]
            summary["images"].append({
                "name": item["name"],
                "peak_reserved_bytes": [row["peak_gpu_memory_bytes"] for row in rows],
                "peak_allocated_bytes": [row["peak_gpu_memory_allocated_bytes"] for row in rows],
                "process_seconds": [row["process_seconds"] for row in rows],
                "official_resize_back": rows[0]["official_resize_back"],
                "prequant_dtype": rows[0]["prequant_dtype"],
                "bitwise_repeatable": len({row["output_tensor_sha256"] for row in rows}) == 1,
            })
        summary["attention_processors"] = runs[0]["manifest"]["attention_processors"]
        summary["vae_key_report"] = {k: v for k, v in runs[0]["manifest"]["vae_key_report"].items()
                                     if k != "checkpoint_top_level_keys"}
        summary["environment"] = runs[0]["manifest"]["environment"]
        summary["first_image_warnings"] = runs[0]["manifest"]["first_image_warnings"]
    (args.work / "probe_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    sys.exit(0 if all(r["exit_code"] == 0 for r in runs) else 1)


if __name__ == "__main__":
    main()
