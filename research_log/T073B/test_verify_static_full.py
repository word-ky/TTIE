"""Mutation self-checks for verify_static_full.py on a tiny synthetic
2-row fixture. No real low/output/reference data is touched here.
"""

import argparse
import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

import verify_static_full as vsf


def build_fixture(root, tamper):
    """Build a synthetic 2-row fixture under root/. tamper selects which
    single field is deliberately wrong: 'none', 'tensor_hash', 'low_sha',
    or 'nan_tensor'. Returns an argparse.Namespace ready for vsf.verify().
    """
    low_dir = root / "low"
    outputs_dir = root / "outputs"
    low_dir.mkdir()
    outputs_dir.mkdir()

    names = ["fakeA_UHD_LL.JPG", "fakeB_UHD_LL.JPG"]
    items = []
    rows = []
    first_row_tensor_sha256 = None

    for index, name in enumerate(names):
        low_bytes = f"synthetic-low-content-{index}".encode("utf-8") * 1000
        (low_dir / name).write_bytes(low_bytes)
        real_low_sha = hashlib.sha256(low_bytes).hexdigest()

        tensor = torch.full(vsf.EXPECTED_SHAPE, float(index), dtype=torch.float32)
        if tamper == "nan_tensor" and index == 0:
            tensor[0, 0, 0, 0] = float("nan")
        tensor = tensor.contiguous()
        real_tensor_sha = vsf.thash(tensor)

        stem = Path(name).stem
        out_dir = outputs_dir / stem
        out_dir.mkdir()
        output_path = out_dir / "output.pt.gz"
        with gzip.open(output_path, "wb", compresslevel=1) as stream:
            torch.save(tensor, stream)
        real_file_sha = vsf.sha256_file(output_path)

        recorded_low_sha = real_low_sha
        recorded_tensor_sha = real_tensor_sha
        if tamper == "low_sha" and index == 0:
            recorded_low_sha = "1" * 64
        if tamper == "tensor_hash" and index == 0:
            recorded_tensor_sha = "2" * 64

        row = {
            "low_name": name,
            "low_sha256": recorded_low_sha,
            "shape": vsf.EXPECTED_SHAPE,
            "dtype": vsf.EXPECTED_DTYPE,
            "output_tensor_sha256": recorded_tensor_sha,
            "output_file_sha256": real_file_sha,
            "whole_run_seconds": 1.0 + index,
            "peak_gpu_memory_bytes": 1000 + index,
        }
        (out_dir / "decision.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        rows.append(row)
        items.append({"name": name, "sha256": recorded_low_sha, "bytes": len(low_bytes)})

        if index == 0:
            # The frozen-smoke cross-check target: the *real* tensor hash of
            # row 0, so this test isolates exactly the field under `tamper`
            # instead of also tripping the frozen-smoke check.
            first_row_tensor_sha256 = real_tensor_sha

    manifest = {"method": "synthetic-fixture", "count": len(rows), "rows": rows}
    manifest_path = outputs_dir / "output_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    low_receipt_path = root / "low_receipt.json"
    low_receipt_path.write_text(
        json.dumps({"count": len(items), "files": items}, indent=2), encoding="utf-8"
    )

    return argparse.Namespace(
        low_receipt=low_receipt_path,
        low_dir=low_dir,
        outputs=outputs_dir,
        out=root / "receipt_out.json",
        expected_count=2,
        first_row_name=names[0],
        first_row_tensor_sha256=first_row_tensor_sha256,
        expected_output_manifest_sha256="",
    )


class TestVerifyStaticFull(unittest.TestCase):
    def test_baseline_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = build_fixture(Path(tmp), tamper="none")
            ok, receipt = vsf.verify(args)
            self.assertTrue(ok, receipt["failures"])
            self.assertEqual(receipt["pass_count"], 2)
            self.assertEqual(receipt["fail_count"], 0)
            self.assertEqual(receipt["failures"], [])
            self.assertTrue(receipt["first_row_frozen_smoke_verified"])

    def test_tampered_tensor_hash_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = build_fixture(Path(tmp), tamper="tensor_hash")
            ok, receipt = vsf.verify(args)
            self.assertFalse(ok)
            self.assertTrue(
                any("output_tensor_sha256 mismatch" in f for f in receipt["failures"]),
                receipt["failures"],
            )

    def test_wrong_low_sha_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = build_fixture(Path(tmp), tamper="low_sha")
            ok, receipt = vsf.verify(args)
            self.assertFalse(ok)
            self.assertTrue(
                any("low image sha256" in f for f in receipt["failures"]),
                receipt["failures"],
            )

    def test_nan_tensor_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = build_fixture(Path(tmp), tamper="nan_tensor")
            ok, receipt = vsf.verify(args)
            self.assertFalse(ok)
            self.assertTrue(
                any("non-finite" in f for f in receipt["failures"]),
                receipt["failures"],
            )


if __name__ == "__main__":
    unittest.main()
