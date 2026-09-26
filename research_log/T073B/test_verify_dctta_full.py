"""Mutation self-checks for verify_dctta_full.py on a tiny synthetic
2-row fixture. No real low/output/reference/adapted-state data is touched
here. Tensor geometry is patched to a tiny shape via the module's
EXPECTED_SHAPE constant so the fixture stays fast.
"""

import argparse
import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

import verify_dctta_full as vdf

TINY_SHAPE = [1, 3, 4, 4]
NAMES = ["fakeA_UHD_LL.JPG", "fakeB_UHD_LL.JPG"]


def build_fixture(root, tamper):
    """Build a synthetic 2-row DCTTA fixture under root/. tamper selects
    which single aspect is deliberately wrong: 'none', 'adapted_state_tamper',
    'not_promotable', 'smoke_method', or 'nan_output'. Returns an
    argparse.Namespace ready for vdf.verify().
    """
    low_dir = root / "low"
    outputs_dir = root / "outputs"
    low_dir.mkdir()
    outputs_dir.mkdir()

    items = []
    rows = []
    fisher_order = []
    adaptation_order = []

    for index, name in enumerate(NAMES):
        low_bytes = f"synthetic-low-content-{index}".encode("utf-8") * 1000
        (low_dir / name).write_bytes(low_bytes)
        real_low_sha = hashlib.sha256(low_bytes).hexdigest()

        tensor = torch.full(TINY_SHAPE, float(index), dtype=torch.float32)
        if tamper == "nan_output" and index == 0:
            tensor[0, 0, 0, 0] = float("nan")
        tensor = tensor.contiguous()
        real_tensor_sha = vdf.thash(tensor)

        stem = Path(name).stem
        out_dir = outputs_dir / stem
        out_dir.mkdir()
        output_path = out_dir / "output.pt.gz"
        with gzip.open(output_path, "wb", compresslevel=1) as stream:
            torch.save(tensor, stream)
        real_file_sha = vdf.sha256_file(output_path)

        row = {
            "low_name": name,
            "low_sha256": real_low_sha,
            "shape": TINY_SHAPE,
            "dtype": "torch.float32",
            "output_tensor_sha256": real_tensor_sha,
            "output_file_sha256": real_file_sha,
            "whole_run_seconds": 1.0 + index,
            "peak_gpu_memory_bytes": 1000 + index,
        }
        (out_dir / "decision.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        rows.append(row)
        items.append({"name": name, "sha256": real_low_sha, "bytes": len(low_bytes)})

        patch_sha = hashlib.sha256(f"patch-{index}".encode()).hexdigest()
        fisher_order.append([name, patch_sha])
        adaptation_order.append([name, patch_sha])

    # Real runs shuffle the fisher and adaptation passes independently; keep
    # them in different orders here too so a fisher/adaptation mixup would
    # be caught by the exact-equality check.
    fisher_order = list(reversed(fisher_order))

    low_receipt_path = root / "low_receipt.json"
    low_receipt_path.write_text(
        json.dumps({"count": len(items), "files": items}, indent=2), encoding="utf-8"
    )
    low_receipt_sha256 = vdf.sha256_file(low_receipt_path)

    order_doc = {"fisher": fisher_order, "adaptation": adaptation_order}
    expected_order_path = root / "adaptation_order.json"
    expected_order_path.write_text(json.dumps(order_doc, indent=2), encoding="utf-8")
    real_order_sha = vdf.sha256_file(expected_order_path)

    # Two small trainable tensors, exactly one changed element each.
    source_state = {
        "encoder.prompt.weight": torch.zeros(3, 3),
        "decoder.head.bias": torch.zeros(4),
    }
    adapted_state = {
        "encoder.prompt.weight": torch.ones(3, 3),
        "decoder.head.bias": torch.full((4,), 2.0),
    }
    trainable_names = list(source_state)
    state_file = outputs_dir / "adapted_state.pt"
    torch.save(adapted_state, state_file)
    adapted_state_sha256 = vdf.state_sha256(adapted_state)
    recorded_file_sha256 = vdf.sha256_file(state_file)
    if tamper == "adapted_state_tamper":
        # Overwrite the on-disk file *after* recording its correct hash, so
        # the manifest's claims no longer match what is actually on disk.
        tampered = {k: v.clone() for k, v in adapted_state.items()}
        tampered["decoder.head.bias"][0] = 99.0
        torch.save(tampered, state_file)

    changed = {
        key: int((adapted_state[key] != source_state[key]).sum())
        for key in source_state
    }
    changed_names = [key for key, count in changed.items() if count]
    parameter_updates = {
        "trainable_tensor_count": len(trainable_names),
        "trainable_elements": sum(v.numel() for v in source_state.values()),
        "trainable_bytes": sum(v.numel() * v.element_size() for v in source_state.values()),
        "changed_tensor_count": len(changed_names),
        "changed_elements": sum(changed.values()),
        "changed_bytes": sum(changed[k] * adapted_state[k].element_size() for k in changed_names),
        "changed_outside_trainable": [],
        "changed_tensor_names": changed_names,
    }

    manifest = {
        "checkpoint_sha256": vdf.EXPECTED_CHECKPOINT_SHA256,
        "low_receipt_sha256": low_receipt_sha256,
        "adaptation_scope": vdf.EXPECTED_ADAPTATION_SCOPE,
        "method": "PromptIR+DCTTA-five-task-domain-level-native-schedule",
        "promotable": True,
        "count": len(rows),
        "expected_order_sha256": real_order_sha,
        "fisher_order": fisher_order,
        "adaptation_order": adaptation_order,
        "adaptation_seconds": 12.3,
        "adaptation_peak_gpu_memory_bytes": 1234,
        "inference_seconds": 5.6,
        "inference_peak_gpu_memory_bytes": max(r["peak_gpu_memory_bytes"] for r in rows),
        "adapted_state_sha256": adapted_state_sha256,
        "adapted_state_file_sha256": recorded_file_sha256,
        "parameter_updates": parameter_updates,
        "reference_reads": 0,
        "metrics": 0,
        "inference_numerics": {"matmul_allow_tf32": False},
        "rows": rows,
    }

    if tamper == "not_promotable":
        manifest["promotable"] = False
    if tamper == "smoke_method":
        manifest["method"] = manifest["method"] + "-SMOKE-NONFINAL"

    manifest_path = outputs_dir / "output_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    args = argparse.Namespace(
        low_receipt=low_receipt_path,
        low_dir=low_dir,
        outputs=outputs_dir,
        expected_order=expected_order_path,
        out=root / "receipt_out.json",
        expected_count=2,
    )
    return args, real_order_sha


class TestVerifyDcttaFull(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._orig_shape = vdf.EXPECTED_SHAPE
        vdf.EXPECTED_SHAPE = TINY_SHAPE

    @classmethod
    def tearDownClass(cls):
        vdf.EXPECTED_SHAPE = cls._orig_shape

    def setUp(self):
        self._orig_order_sha = vdf.EXPECTED_ORDER_SHA256_DEFAULT

    def tearDown(self):
        vdf.EXPECTED_ORDER_SHA256_DEFAULT = self._orig_order_sha

    def test_baseline_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, real_order_sha = build_fixture(Path(tmp), tamper="none")
            vdf.EXPECTED_ORDER_SHA256_DEFAULT = real_order_sha
            ok, receipt = vdf.verify(args)
            self.assertTrue(ok, receipt["failures"])
            self.assertEqual(receipt["pass_count"], 2)
            self.assertEqual(receipt["fail_count"], 0)
            self.assertEqual(receipt["failures"], [])
            self.assertTrue(receipt["parameter_updates_ok"])
            self.assertTrue(receipt["adapted_state_finite"])

    def test_tampered_adapted_state_file_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, real_order_sha = build_fixture(Path(tmp), tamper="adapted_state_tamper")
            vdf.EXPECTED_ORDER_SHA256_DEFAULT = real_order_sha
            ok, receipt = vdf.verify(args)
            self.assertFalse(ok)
            self.assertTrue(
                any("adapted_state.pt file SHA256 does not match manifest" in f for f in receipt["failures"]),
                receipt["failures"],
            )

    def test_wrong_expected_order_sha_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, real_order_sha = build_fixture(Path(tmp), tamper="none")
            vdf.EXPECTED_ORDER_SHA256_DEFAULT = "f" * 64  # deliberately wrong sealed constant
            ok, receipt = vdf.verify(args)
            self.assertFalse(ok)
            self.assertTrue(
                any("sealed order SHA256" in f for f in receipt["failures"]),
                receipt["failures"],
            )

    def test_not_promotable_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, real_order_sha = build_fixture(Path(tmp), tamper="not_promotable")
            vdf.EXPECTED_ORDER_SHA256_DEFAULT = real_order_sha
            ok, receipt = vdf.verify(args)
            self.assertFalse(ok)
            self.assertTrue(
                any("promotable is not True" in f for f in receipt["failures"]),
                receipt["failures"],
            )

    def test_smoke_method_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, real_order_sha = build_fixture(Path(tmp), tamper="smoke_method")
            vdf.EXPECTED_ORDER_SHA256_DEFAULT = real_order_sha
            ok, receipt = vdf.verify(args)
            self.assertFalse(ok)
            self.assertTrue(
                any("method string contains SMOKE" in f for f in receipt["failures"]),
                receipt["failures"],
            )

    def test_nan_output_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, real_order_sha = build_fixture(Path(tmp), tamper="nan_output")
            vdf.EXPECTED_ORDER_SHA256_DEFAULT = real_order_sha
            ok, receipt = vdf.verify(args)
            self.assertFalse(ok)
            self.assertTrue(
                any("non-finite" in f for f in receipt["failures"]),
                receipt["failures"],
            )


if __name__ == "__main__":
    unittest.main()
