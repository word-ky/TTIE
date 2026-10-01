"""Independently verify dataset-generic QuadPrior low-only outputs (T073-E).

Never enumerates, opens or references any clean/GT path and computes no
PSNR/SSIM or other quality metric. It re-hashes low images and every output
artifact, cross-checks manifest rows against decision.json and the frozen low
receipt, checks geometry against the low image's own cv2 decode (the official
decoder), checks that each native float lies inside the truncation bin of the
official PNG written by test.py, and recomputes the bilinear map-back.
"""

import argparse
import gzip
import hashlib
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

EXPECTED_DTYPE = "torch.float32"
BIN_SLACK = 1e-3
MAP_BACK_TOLERANCE = 1e-6


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def thash(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def self_sha256_lf():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_gz(path):
    with gzip.open(path, "rb") as stream:
        return torch.load(stream, map_location="cpu", weights_only=True)


def official_native_hw(height, width, resolution=512):
    """annotator/util.py:28-37 arithmetic."""
    h, w = float(height), float(width)
    k = float(resolution) / min(h, w)
    return [int(np.round(h * k / 64.0)) * 64, int(np.round(w * k / 64.0)) * 64]


def bin_violations(native_rgb, official_bgr):
    scaled = np.clip(native_rgb[0].numpy().transpose(1, 2, 0).astype(np.float64) * 255.0, 0.0, 255.0)
    u8 = official_bgr[..., ::-1].astype(np.float64)
    low_bad = scaled < u8 - BIN_SLACK
    high_bad = (scaled >= u8 + 1.0 + BIN_SLACK) & (u8 < 255)
    return int(low_bad.sum() + high_bad.sum())


def map_back(native_tensor, low_hw):
    """Must match run_quadprior_batch.map_back."""
    height, width = low_hw
    wide = np.clip(native_tensor[0].numpy().transpose(1, 2, 0).astype(np.float64), 0.0, 1.0)
    return cv2.resize(wide, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--low-receipt", type=Path, required=True)
    parser.add_argument("--low-dir", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--expected-coco-sha256", required=True)
    parser.add_argument("--expected-vae-sha256", required=True)
    parser.add_argument("--expected-control-sha256", required=True)
    parser.add_argument("--expected-output-manifest-sha256", default="",
                        help="the run's logged OUTPUT_MANIFEST_SHA256; '' skips (tests only)")
    parser.add_argument("--first-row-tensor-sha256", default="",
                        help="frozen smoke tensor hash of the first receipt row; '' skips")
    parser.add_argument("--require-promotable", action="store_true")
    return parser


def verify(args):
    failures = []
    fail = failures.append
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    torch.set_num_threads(4)

    receipt_sha = sha256_file(args.low_receipt)
    expected_files = json.loads(args.low_receipt.read_bytes()).get("files", [])
    expected_names = [item["name"] for item in expected_files]
    if len(expected_files) != args.expected_count:
        fail(f"low receipt does not have {args.expected_count} files: {len(expected_files)}")
    if len(set(expected_names)) != len(expected_names):
        fail("low receipt contains duplicate names")

    manifest, manifest_sha, rows = {}, None, []
    manifest_path = args.outputs / "output_manifest.json"
    if not manifest_path.exists():
        fail(f"output_manifest.json missing at {manifest_path}")
    else:
        manifest_sha = sha256_file(manifest_path)
        if args.expected_output_manifest_sha256 and manifest_sha != args.expected_output_manifest_sha256:
            fail(f"output_manifest.json SHA256 differs from the logged value: {manifest_sha}")
        manifest = json.loads(manifest_path.read_bytes())
        rows = manifest.get("rows", [])

    for key, value in {"count": args.expected_count, "reference_reads": 0, "metrics": 0,
                       "model_state_unchanged": True, "low_receipt_sha256": receipt_sha}.items():
        if manifest.get(key) != value:
            fail(f"manifest {key} != {value!r}: {manifest.get(key)!r}")
    weights = manifest.get("weights") or {}
    for label, expected in (("coco_final", args.expected_coco_sha256),
                            ("vae_main_epoch00_step7000", args.expected_vae_sha256),
                            ("control_sd15_ini", args.expected_control_sha256)):
        if (weights.get(label) or {}).get("sha256") != expected:
            fail(f"manifest weight {label} sha256 mismatch")
    official_args = manifest.get("official_args") or {}
    if official_args.get("use_float16") is not True or official_args.get("save_memory") is not False:
        fail("manifest official args are not the official defaults")
    if args.require_promotable and (manifest.get("promotable") is not True
                                    or manifest.get("declared_count") != args.expected_count):
        fail("manifest is not promotable for the declared count")
    if len(rows) != args.expected_count:
        fail(f"manifest rows length != {args.expected_count}: {len(rows)}")
    names = [row.get("low_name") for row in rows]
    if names != expected_names:
        fail("manifest row order/names do not exactly match the frozen low receipt")
    if sorted(manifest.get("official_processing_order") or []) != sorted(expected_names):
        fail("official processing order is not a permutation of the receipt")

    official_dir = args.outputs / "official_output"
    if not official_dir.is_dir():
        fail("official_output directory missing")
    elif sorted(p.name for p in official_dir.iterdir()) != sorted(Path(n).stem + ".png" for n in expected_names):
        fail("official_output does not contain exactly one PNG per receipt name")

    pass_count, runtimes, peaks, first_verified = 0, [], [], False
    for index, (item, row) in enumerate(zip(expected_files, rows)):
        name = item["name"]
        before = len(failures)
        try:
            if row.get("low_name") != name:
                fail(f"{name}: manifest row low_name mismatch")
            low_path = args.low_dir / name
            low_sha = sha256_file(low_path)
            if low_sha != item["sha256"]:
                fail(f"{name}: low sha256 does not match frozen receipt")
            if low_sha != row.get("low_sha256"):
                fail(f"{name}: low sha256 does not match manifest row")
            low = cv2.imread(str(low_path))
            height, width = low.shape[:2]
            del low
            native_hw = official_native_hw(height, width)
            if row.get("low_hw") != [height, width] or row.get("shape") != [1, 3, height, width]:
                fail(f"{name}: recorded geometry differs from the low image ({height}x{width})")
            if row.get("native_shape") != [1, 3, *native_hw] or row.get("dtype") != EXPECTED_DTYPE:
                fail(f"{name}: recorded native shape/dtype mismatch")
            if row.get("official_uint8_match") is not True:
                fail(f"{name}: runner provenance flag not true")
            out_dir = args.outputs / Path(name).stem
            decision_path = out_dir / "decision.json"
            if not decision_path.exists() or json.loads(decision_path.read_bytes()) != row:
                fail(f"{name}: decision.json missing or differs from manifest row")

            tensors = {}
            for label, file_name, shape, hash_key, file_key in (
                    ("output", "output.pt.gz", [1, 3, height, width], "output_tensor_sha256", "output_file_sha256"),
                    ("native", "native_output.pt.gz", [1, 3, *native_hw], "native_tensor_sha256",
                     "native_file_sha256")):
                path = out_dir / file_name
                if not path.exists() or sha256_file(path) != row.get(file_key):
                    fail(f"{name}: {file_name} missing or {file_key} mismatch")
                    continue
                tensor = load_gz(path)
                if list(tensor.shape) != shape or str(tensor.dtype) != EXPECTED_DTYPE:
                    fail(f"{name}: {label} tensor shape/dtype mismatch {list(tensor.shape)} {tensor.dtype}")
                elif not torch.isfinite(tensor).all():
                    fail(f"{name}: {label} tensor contains non-finite values")
                elif thash(tensor) != row.get(hash_key):
                    fail(f"{name}: {hash_key} mismatch")
                else:
                    tensors[label] = tensor
            if index == 0 and args.first_row_tensor_sha256 and "output" in tensors:
                if thash(tensors["output"]) == args.first_row_tensor_sha256:
                    first_verified = True
                else:
                    fail(f"{name}: frozen smoke tensor sha256 mismatch")
            if "output" in tensors and "native" in tensors:
                diff = np.abs(map_back(tensors["native"], (height, width))
                              - tensors["output"][0].numpy().transpose(1, 2, 0)).max()
                if diff > MAP_BACK_TOLERANCE:
                    fail(f"{name}: map-back recomputation differs (max abs {diff})")

            png_path = args.outputs / str(row.get("official_output_file", ""))
            if not png_path.is_file() or sha256_file(png_path) != row.get("official_output_file_sha256"):
                fail(f"{name}: official PNG missing or hash mismatch")
            else:
                official = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
                if official is None or list(official.shape) != [*native_hw, 3] or official.dtype != np.uint8:
                    fail(f"{name}: official PNG geometry mismatch")
                else:
                    if hashlib.sha256(np.ascontiguousarray(official).tobytes()).hexdigest() != row.get(
                            "official_uint8_bgr_sha256"):
                        fail(f"{name}: official uint8 array hash mismatch")
                    if "native" in tensors:
                        bad = bin_violations(tensors["native"], official)
                        if bad:
                            fail(f"{name}: {bad} float values outside the official uint8 truncation bin")
            if isinstance(row.get("whole_run_seconds"), (int, float)):
                runtimes.append(row["whole_run_seconds"])
            if isinstance(row.get("peak_gpu_memory_bytes"), (int, float)):
                peaks.append(row["peak_gpu_memory_bytes"])
        except Exception as exc:  # noqa: BLE001 - one row must not abort the rest
            fail(f"{name}: unexpected verification error: {exc}")
        if len(failures) == before:
            pass_count += 1

    if args.first_row_tensor_sha256 and not first_verified:
        fail("frozen smoke tensor hash for the first receipt row did not verify")

    def summary(values):
        if not values:
            return {"n": 0, "min": None, "median": None, "max": None}
        return {"n": len(values), "min": min(values), "median": statistics.median(values), "max": max(values)}

    ok = not failures
    return ok, {
        "classification": "QUADPRIOR_OUTPUTS_VERIFIED" if ok else "QUADPRIOR_OUTPUTS_FAILED",
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - started,
        "verifier_sha256_lf": self_sha256_lf(),
        "low_receipt_sha256": receipt_sha,
        "outputs_dir": str(args.outputs),
        "output_manifest_sha256": manifest_sha,
        "expected_count": args.expected_count,
        "observed_row_count": len(rows),
        "pass_count": pass_count,
        "failures": failures,
        "manifest_promotable": manifest.get("promotable"),
        "manifest_runner_sha256_lf": manifest.get("runner_sha256_lf"),
        "first_row_frozen_smoke_verified": first_verified,
        "whole_run_seconds": summary(runtimes),
        "peak_gpu_memory_bytes": summary(peaks),
        "reference_reads": 0,
        "metrics": 0,
    }


def main():
    args = build_parser().parse_args()
    ok, receipt = verify(args)
    args.out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in receipt.items() if k != "failures"}, indent=2))
    if not ok:
        print("FAILURES:", *receipt["failures"], sep="\n  - ", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
