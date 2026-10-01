"""Independently verify the completed PromptIR+DCTTA UHD-LL full-run
(150-image) low-only outputs, including the domain-level adaptation phase.

This never opens, enumerates, or references any UHD-LL clean/normal-light/GT
path, and never computes PSNR/SSIM or any quality metric. It only re-hashes
low images, output tensors/files, the adapted weights, and cross-checks the
run's own manifest and per-row decision.json records against a frozen
150-name low-image receipt and the sealed adaptation-order file. It never
imports the runner (run_dctta_native_batch.py); every hash algorithm it
relies on is reimplemented here from the runner's own source.
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
EXPECTED_CHECKPOINT_SHA256 = (
    "206baf0dd10f636f025b33b5ee7eb63a353fcbf4d50858b62f9480a6d4be9d4a"
)
EXPECTED_ADAPTATION_SCOPE = "all_target_low_domain_level"
EXPECTED_ORDER_SHA256_DEFAULT = (
    "40c2e937ebf97c0a4c67ce6b40cfc632699956e502fb79b87250112181623ee1"
)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def thash(tensor):
    """Same output-tensor hash formula the runner used: sha256 of contiguous
    bytes (run_dctta_native_batch.main: hashlib.sha256(output_cpu.numpy().tobytes()))."""
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def state_sha256(state):
    """Same state hash formula as run_dctta_native_batch.state_sha256:
    for each key (in dict order) hash 'key|dtype|shape|' then the raw bytes."""
    digest = hashlib.sha256()
    for key, value in state.items():
        value = value.detach().cpu().contiguous()
        digest.update(f"{key}|{value.dtype}|{tuple(value.shape)}|".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


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
    parser.add_argument("--expected-order", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=EXPECTED_COUNT_DEFAULT)
    return parser


def check_order_coverage(label, entries, expected_names, count, fail):
    if not isinstance(entries, list) or len(entries) != count:
        fail(f"{label} does not have {count} entries: {len(entries) if isinstance(entries, list) else entries}")
        return
    names_only = [item[0] for item in entries]
    if set(names_only) != set(expected_names):
        fail(f"{label} does not cover exactly the {count} expected names")


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

    expected_order_file_sha256 = sha256_file(args.expected_order)
    expected_order = json.loads(args.expected_order.read_bytes())

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
            manifest = json.loads(manifest_path.read_bytes())
            rows = manifest.get("rows", [])

    # --- manifest-level checks -------------------------------------------------
    if manifest:
        if manifest.get("count") != args.expected_count:
            fail(f"manifest count != {args.expected_count}: {manifest.get('count')}")
        if len(rows) != args.expected_count:
            fail(f"manifest rows length != {args.expected_count}: {len(rows)}")

        if manifest.get("promotable") is not True:
            fail(f"manifest promotable is not True: {manifest.get('promotable')}")
        if manifest.get("adaptation_scope") != EXPECTED_ADAPTATION_SCOPE:
            fail(f"manifest adaptation_scope mismatch: {manifest.get('adaptation_scope')}")
        method = manifest.get("method", "")
        if "SMOKE" in method:
            fail(f"manifest method string contains SMOKE: {method}")
        if manifest.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
            fail(f"manifest checkpoint_sha256 mismatch: {manifest.get('checkpoint_sha256')}")
        if manifest.get("low_receipt_sha256") != low_receipt_sha256:
            fail(
                "manifest low_receipt_sha256 does not match the recomputed "
                f"receipt file SHA256: {manifest.get('low_receipt_sha256')}"
            )

        manifest_expected_order_sha256 = manifest.get("expected_order_sha256")
        if manifest_expected_order_sha256 != EXPECTED_ORDER_SHA256_DEFAULT:
            fail(
                "manifest expected_order_sha256 does not match the sealed "
                f"order SHA256: {manifest_expected_order_sha256}"
            )
        if expected_order_file_sha256 != EXPECTED_ORDER_SHA256_DEFAULT:
            fail(
                "--expected-order file SHA256 does not match the sealed "
                f"order SHA256: {expected_order_file_sha256}"
            )
        if manifest_expected_order_sha256 != expected_order_file_sha256:
            fail("manifest expected_order_sha256 does not match --expected-order file SHA256")

        manifest_fisher_order = manifest.get("fisher_order")
        manifest_adaptation_order = manifest.get("adaptation_order")
        if manifest_fisher_order != expected_order.get("fisher"):
            fail("manifest fisher_order does not exactly equal the order file's fisher order")
        if manifest_adaptation_order != expected_order.get("adaptation"):
            fail("manifest adaptation_order does not exactly equal the order file's adaptation order")

        check_order_coverage(
            "order file fisher order", expected_order.get("fisher", []),
            expected_names, args.expected_count, fail,
        )
        check_order_coverage(
            "order file adaptation order", expected_order.get("adaptation", []),
            expected_names, args.expected_count, fail,
        )
        check_order_coverage(
            "manifest fisher_order", manifest_fisher_order or [],
            expected_names, args.expected_count, fail,
        )
        check_order_coverage(
            "manifest adaptation_order", manifest_adaptation_order or [],
            expected_names, args.expected_count, fail,
        )

        if manifest.get("reference_reads") != 0:
            fail(f"manifest reference_reads != 0: {manifest.get('reference_reads')}")
        if manifest.get("metrics") != 0:
            fail(f"manifest metrics != 0: {manifest.get('metrics')}")

        inference_numerics = manifest.get("inference_numerics", {})
        if inference_numerics.get("matmul_allow_tf32") is not False:
            fail(
                "manifest inference_numerics.matmul_allow_tf32 is not False: "
                f"{inference_numerics.get('matmul_allow_tf32')}"
            )

    manifest_names = [row.get("low_name") for row in rows]
    if manifest_names != expected_names:
        fail(
            "manifest row order/names do not exactly match the frozen "
            "low-image receipt (order matters)"
        )
    if len(manifest_names) != len(set(manifest_names)):
        fail("manifest contains duplicate low_name entries")

    # --- adapted state -----------------------------------------------------
    adapted_state_path = args.outputs / "adapted_state.pt"
    adapted_state_file_sha256 = None
    adapted_state_sha256 = None
    adapted_state_finite = None
    parameter_updates_ok = None
    if not adapted_state_path.exists():
        fail(f"adapted_state.pt missing at {adapted_state_path}")
    else:
        adapted_state_file_sha256 = sha256_file(adapted_state_path)
        expected_file_sha = manifest.get("adapted_state_file_sha256")
        if adapted_state_file_sha256 != expected_file_sha:
            fail(
                "adapted_state.pt file SHA256 does not match manifest "
                f"adapted_state_file_sha256: recomputed {adapted_state_file_sha256}"
            )
        try:
            state = torch.load(adapted_state_path, map_location="cpu", weights_only=True)
        except Exception as exc:  # noqa: BLE001 - report and continue
            fail(f"adapted_state.pt failed to load: {exc}")
            state = None

        if state is not None:
            adapted_state_finite = all(
                torch.isfinite(value).all().item()
                for value in state.values()
                if torch.is_tensor(value) and value.is_floating_point()
            )
            if not adapted_state_finite:
                fail("adapted_state.pt contains non-finite floating tensors")

            adapted_state_sha256 = state_sha256(state)
            if adapted_state_sha256 != manifest.get("adapted_state_sha256"):
                fail(
                    "recomputed adapted_state_sha256 does not match manifest: "
                    f"recomputed {adapted_state_sha256}"
                )

        parameter_updates = manifest.get("parameter_updates", {})
        changed_tensor_count = parameter_updates.get("changed_tensor_count", 0)
        changed_elements = parameter_updates.get("changed_elements", 0)
        changed_outside_trainable = parameter_updates.get("changed_outside_trainable")
        parameter_updates_ok = (
            isinstance(changed_tensor_count, int) and changed_tensor_count > 0
            and isinstance(changed_elements, int) and changed_elements > 0
            and changed_outside_trainable == []
        )
        if not parameter_updates_ok:
            fail(
                "manifest parameter_updates is not a valid promotable update: "
                f"changed_tensor_count={changed_tensor_count} "
                f"changed_elements={changed_elements} "
                f"changed_outside_trainable={changed_outside_trainable}"
            )

    # --- per-row checks ------------------------------------------------------
    pass_count = 0
    fail_count = 0
    runtimes = []
    peak_mem_bytes = []

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
            "UHDLL_DCTTA_FULL_OUTPUTS_VERIFIED" if ok else "UHDLL_DCTTA_FULL_OUTPUTS_FAILED"
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
        "expected_order_path": str(args.expected_order),
        "expected_order_file_sha256": expected_order_file_sha256,
        "sealed_expected_order_sha256": EXPECTED_ORDER_SHA256_DEFAULT,
        "manifest_expected_order_sha256": manifest.get("expected_order_sha256"),
        "expected_count": args.expected_count,
        "observed_row_count": len(rows),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "failures": failures,
        "manifest_checkpoint_sha256": manifest.get("checkpoint_sha256"),
        "manifest_promotable": manifest.get("promotable"),
        "manifest_adaptation_scope": manifest.get("adaptation_scope"),
        "manifest_method": manifest.get("method"),
        "manifest_reference_reads": manifest.get("reference_reads"),
        "manifest_metrics": manifest.get("metrics"),
        "manifest_inference_numerics": manifest.get("inference_numerics"),
        "adapted_state_path": str(adapted_state_path),
        "adapted_state_file_sha256": adapted_state_file_sha256,
        "adapted_state_sha256": adapted_state_sha256,
        "adapted_state_finite": adapted_state_finite,
        "manifest_adapted_state_sha256": manifest.get("adapted_state_sha256"),
        "manifest_adapted_state_file_sha256": manifest.get("adapted_state_file_sha256"),
        "parameter_updates": manifest.get("parameter_updates"),
        "parameter_updates_ok": parameter_updates_ok,
        "adaptation_seconds": manifest.get("adaptation_seconds"),
        "adaptation_peak_gpu_memory_bytes": manifest.get("adaptation_peak_gpu_memory_bytes"),
        "inference_seconds": manifest.get("inference_seconds"),
        "inference_peak_gpu_memory_bytes_manifest": manifest.get("inference_peak_gpu_memory_bytes"),
        "inference_per_image_whole_run_seconds": summary(runtimes),
        "inference_per_image_peak_gpu_memory_bytes": summary(peak_mem_bytes),
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
