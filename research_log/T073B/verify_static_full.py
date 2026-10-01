"""Independently verify the completed PromptIR static UHD-LL full-run
(150-image) low-only outputs.

This never opens, enumerates, or references any UHD-LL clean/normal-light/GT
path, and never computes PSNR/SSIM or any quality metric. It only re-hashes
low images, output tensors/files, and cross-checks the run's own manifest and
per-row decision.json records against a frozen 150-name low-image receipt.
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

import torch

EXPECTED_COUNT_DEFAULT = 150
EXPECTED_SHAPE = [1, 3, 2160, 3840]
EXPECTED_DTYPE = "torch.float32"
EXPECTED_FIRST_ROW_NAME_DEFAULT = "1003_UHD_LL.JPG"
EXPECTED_FIRST_ROW_TENSOR_SHA256_DEFAULT = (
    "edd0d508e737f5460cb93e2cc153479b3f72e9948ab9ebb19d75ca1c1d00db2b"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "206baf0dd10f636f025b33b5ee7eb63a353fcbf4d50858b62f9480a6d4be9d4a"
)
EXPECTED_OUTPUT_MANIFEST_SHA256_DEFAULT = (
    "be2ab4e9fa919d0a9ffc5764135ede17be0e8d5bc353594c983946c8f8a24913"
)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def thash(tensor):
    """Same tensor-hash formula the runner used: sha256 of contiguous bytes."""
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def self_sha256_lf():
    """SHA256 of this script's own source, with CRLF normalized to LF."""
    src = Path(__file__).read_bytes()
    normalized = src.replace(b"\r\n", b"\n")
    return hashlib.sha256(normalized).hexdigest()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low-receipt", type=Path, required=True)
    parser.add_argument("--low-dir", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=EXPECTED_COUNT_DEFAULT)
    parser.add_argument(
        "--first-row-name", default=EXPECTED_FIRST_ROW_NAME_DEFAULT
    )
    parser.add_argument(
        "--first-row-tensor-sha256",
        default=EXPECTED_FIRST_ROW_TENSOR_SHA256_DEFAULT,
    )
    parser.add_argument(
        "--expected-output-manifest-sha256",
        default=EXPECTED_OUTPUT_MANIFEST_SHA256_DEFAULT,
        help="Pass '' to skip this cross-check (used by mutation tests).",
    )
    return parser


def verify(args):
    """Run the independent verification. Returns (ok, receipt_dict)."""
    failures = []

    def fail(msg):
        failures.append(msg)

    started_wall = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()

    torch.set_num_threads(4)

    low_receipt_sha256 = sha256_file(args.low_receipt)
    receipt = json.loads(args.low_receipt.read_bytes())
    expected_files = receipt.get("files", [])
    expected_names = [item["name"] for item in expected_files]

    if len(expected_files) != args.expected_count:
        fail(
            f"low receipt does not have {args.expected_count} files: "
            f"{len(expected_files)}"
        )
    if len(expected_names) != len(set(expected_names)):
        fail("low receipt contains duplicate names")

    if not args.outputs.exists():
        fail(f"outputs directory missing: {args.outputs}")
        manifest = {}
        manifest_sha256 = None
        rows = []
    else:
        manifest_path = args.outputs / "output_manifest.json"
        if not manifest_path.exists():
            fail(f"output_manifest.json missing at {manifest_path}")
            manifest = {}
            manifest_sha256 = None
            rows = []
        else:
            manifest_sha256 = sha256_file(manifest_path)
            if args.expected_output_manifest_sha256:
                if manifest_sha256 != args.expected_output_manifest_sha256:
                    fail(
                        "output_manifest.json SHA256 does not match the "
                        "run's own logged OUTPUT_MANIFEST_SHA256: "
                        f"{manifest_sha256}"
                    )
            manifest = json.loads(manifest_path.read_bytes())
            rows = manifest.get("rows", [])

    if manifest:
        if manifest.get("count") != args.expected_count:
            fail(f"manifest count != {args.expected_count}: {manifest.get('count')}")
        if len(rows) != args.expected_count:
            fail(f"manifest rows length != {args.expected_count}: {len(rows)}")

    manifest_names = [row.get("low_name") for row in rows]
    if manifest_names != expected_names:
        fail(
            "manifest row order/names do not exactly match the frozen "
            "low-image receipt (order matters)"
        )
    if len(manifest_names) != len(set(manifest_names)):
        fail("manifest contains duplicate low_name entries")

    if "checkpoint_sha256" in manifest:
        if manifest["checkpoint_sha256"] != EXPECTED_CHECKPOINT_SHA256:
            fail(f"manifest checkpoint_sha256 mismatch: {manifest['checkpoint_sha256']}")
    if "model_state_unchanged" in manifest:
        if manifest["model_state_unchanged"] is not True:
            fail(f"manifest model_state_unchanged is not True: {manifest['model_state_unchanged']}")
    if "reference_reads" in manifest:
        if manifest["reference_reads"] != 0:
            fail(f"manifest reference_reads != 0: {manifest['reference_reads']}")
    if "metrics" in manifest:
        if manifest["metrics"] != 0:
            fail(f"manifest metrics != 0: {manifest['metrics']}")

    pass_count = 0
    fail_count = 0
    runtimes = []
    peak_mem_bytes = []
    first_row_verified = False

    for item, row in zip(expected_files, rows):
        name = item["name"]
        row_ok = True
        try:
            if row.get("low_name") != name:
                fail(f"{name}: manifest row low_name mismatch")
                row_ok = False

            low_path = args.low_dir / name
            if not low_path.exists():
                fail(f"{name}: low image file missing at {low_path}")
                row_ok = False
            else:
                low_sha = sha256_file(low_path)
                if low_sha != item["sha256"]:
                    fail(f"{name}: low image sha256 does not match frozen receipt")
                    row_ok = False
                if low_sha != row.get("low_sha256"):
                    fail(f"{name}: low image sha256 does not match manifest row")
                    row_ok = False

            stem = Path(name).stem
            out_dir = args.outputs / stem
            decision_path = out_dir / "decision.json"
            output_path = out_dir / "output.pt.gz"

            decision = None
            if not decision_path.exists():
                fail(f"{name}: decision.json missing")
                row_ok = False
            else:
                decision = json.loads(decision_path.read_bytes())
                if decision != row:
                    fail(f"{name}: decision.json does not agree with manifest row")
                    row_ok = False

            if not output_path.exists():
                fail(f"{name}: output.pt.gz missing")
                row_ok = False
            else:
                file_sha = sha256_file(output_path)
                if file_sha != row.get("output_file_sha256"):
                    fail(f"{name}: output_file_sha256 mismatch (recomputed {file_sha})")
                    row_ok = False
                if decision is not None and file_sha != decision.get("output_file_sha256"):
                    fail(f"{name}: output_file_sha256 mismatch vs decision.json")
                    row_ok = False

                tensor = None
                try:
                    with gzip.open(output_path, "rb") as stream:
                        tensor = torch.load(stream, map_location="cpu", weights_only=True)
                except Exception as exc:  # noqa: BLE001 - report and continue
                    fail(f"{name}: failed to load tensor: {exc}")
                    row_ok = False

                if tensor is not None:
                    if list(tensor.shape) != EXPECTED_SHAPE:
                        fail(f"{name}: tensor shape mismatch: {list(tensor.shape)}")
                        row_ok = False
                    if str(tensor.dtype) != EXPECTED_DTYPE:
                        fail(f"{name}: tensor dtype mismatch: {tensor.dtype}")
                        row_ok = False
                    if not torch.isfinite(tensor).all():
                        fail(f"{name}: tensor contains non-finite values")
                        row_ok = False
                    tensor_sha = thash(tensor)
                    if tensor_sha != row.get("output_tensor_sha256"):
                        fail(f"{name}: output_tensor_sha256 mismatch (recomputed {tensor_sha})")
                        row_ok = False
                    if name == args.first_row_name:
                        if tensor_sha == args.first_row_tensor_sha256:
                            first_row_verified = True
                        else:
                            fail(
                                f"{name}: frozen smoke tensor sha256 mismatch "
                                f"(recomputed {tensor_sha})"
                            )

            rt = row.get("whole_run_seconds")
            if isinstance(rt, (int, float)):
                runtimes.append(rt)
            pm = row.get("peak_gpu_memory_bytes")
            if isinstance(pm, (int, float)):
                peak_mem_bytes.append(pm)
        except Exception as exc:  # noqa: BLE001 - never let one row abort the rest
            fail(f"{name}: unexpected error during verification: {exc}")
            row_ok = False

        if row_ok:
            pass_count += 1
        else:
            fail_count += 1

    if args.first_row_name not in manifest_names:
        fail(f"frozen first row {args.first_row_name} is absent from the manifest")
    elif manifest_names[0] != args.first_row_name:
        fail(f"first manifest row is not {args.first_row_name}: {manifest_names[0]}")
    if not first_row_verified:
        fail(f"frozen smoke tensor hash for {args.first_row_name} did not verify")

    def summary(values):
        if not values:
            return {"n": 0, "min": None, "median": None, "max": None}
        return {
            "n": len(values),
            "min": min(values),
            "median": statistics.median(values),
            "max": max(values),
        }

    finished_utc = datetime.now(timezone.utc).isoformat()
    wall_seconds = time.perf_counter() - started_wall
    ok = len(failures) == 0

    receipt_out = {
        "classification": (
            "UHDLL_STATIC_FULL_OUTPUTS_VERIFIED" if ok else "UHDLL_STATIC_FULL_OUTPUTS_FAILED"
        ),
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "wall_seconds": wall_seconds,
        "verifier_path": str(Path(__file__).resolve()),
        "verifier_sha256_lf": self_sha256_lf(),
        "low_receipt_path": str(args.low_receipt),
        "low_receipt_sha256": low_receipt_sha256,
        "low_dir": str(args.low_dir),
        "outputs_dir": str(args.outputs),
        "output_manifest_path": str(args.outputs / "output_manifest.json"),
        "output_manifest_sha256": manifest_sha256,
        "expected_output_manifest_sha256": args.expected_output_manifest_sha256 or None,
        "expected_count": args.expected_count,
        "observed_row_count": len(rows),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "failures": failures,
        "manifest_checkpoint_sha256": manifest.get("checkpoint_sha256"),
        "manifest_model_state_unchanged": manifest.get("model_state_unchanged"),
        "manifest_reference_reads": manifest.get("reference_reads"),
        "manifest_metrics": manifest.get("metrics"),
        "first_row_name": manifest_names[0] if manifest_names else None,
        "first_row_frozen_smoke_verified": first_row_verified,
        "whole_run_seconds": summary(runtimes),
        "peak_gpu_memory_bytes": summary(peak_mem_bytes),
        "reference_reads": 0,
        "metrics": 0,
    }
    return ok, receipt_out


def main():
    args = build_parser().parse_args()
    ok, receipt_out = verify(args)
    args.out.write_text(json.dumps(receipt_out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in receipt_out.items() if k != "failures"}, indent=2))
    if not ok:
        print("FAILURES:", file=sys.stderr)
        for item in receipt_out["failures"]:
            print(f"  - {item}", file=sys.stderr)
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
