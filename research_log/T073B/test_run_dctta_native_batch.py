"""Synthetic domain-level DCTTA order, adapted-weight inference and GT-access mutation tests."""

import contextlib
import gzip
import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import run_dctta_native_batch as runner

NAMES = ["a.png", "b.png", "c.png"]
STATIC_ROW_KEYS = {
    "low_name", "low_sha256", "shape", "dtype", "output_tensor_sha256",
    "output_file_sha256", "whole_run_seconds", "peak_gpu_memory_bytes",
}
EVENTS = []


class ToyPromptIR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Conv2d(3, 3, 1)
        with torch.no_grad():
            self.encoder.weight.copy_(torch.eye(3).view(3, 3, 1, 1))
            self.encoder.bias.zero_()

    def forward(self, low, noise_emb=None):
        if low.shape[-1] == 3840:
            EVENTS.append(("infer", None))
        return self.encoder(low)


class FakeAdapted:
    def __init__(self, options, model, optimizer, logger):
        self.model = model

    def compute_fisher(self, loader):
        for names, degraded, surrogate in loader:
            assert torch.equal(degraded, surrogate)
            EVENTS.append(("fisher", names[0]))

    def __call__(self, loader, degraded, names, degeneration_model):
        with torch.no_grad():
            self.model.encoder.bias.add_(0.01)
        EVENTS.append(("adapt", names[0]))
        return self.model(degraded)


class DctaNativeBatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path(os.environ["DCTTA_SOURCE_ROOT"])
        sys.path.insert(0, str(cls.source))
        import tta
        import net.model  # noqa: F401  PromptIR before RDDM, as in the official entrypoint
        from RDDM import net as rddm_net

        cls.tta, cls.rddm_net = tta, rddm_net
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.low_dir, cls.gt_dir = cls.root / "low", cls.root / "gt"
        cls.low_dir.mkdir()
        cls.gt_dir.mkdir()
        y, x = np.indices((2160, 3840), dtype=np.uint32)
        files = []
        for index, name in enumerate(NAMES):
            image = np.stack((x + index, y + 7 * index, x + y), axis=-1) % 256
            Image.fromarray(image.astype(np.uint8)).save(cls.low_dir / name)
            Image.fromarray(255 - image.astype(np.uint8)).save(cls.gt_dir / name)
            files.append({"name": name, "sha256": runner.sha256_file(cls.low_dir / name)})
        cls.receipt = cls.root / "receipt.json"
        cls.receipt.write_text(json.dumps({"count": 3, "files": files}))
        cls.work = cls.root / "work"
        cls.work.mkdir()
        shutil.copy(cls.source / "pretrain" / "wavelet.mat", cls.work / "wavelet.mat")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def run_main(self, out, *extra):
        argv = [
            "run_dctta_native_batch.py", "--source-root", str(self.source),
            "--checkpoint", str(self.receipt), "--low-dir", str(self.low_dir),
            "--low-receipt", str(self.receipt), "--out", str(self.root / out),
            "--num-workers", "0", *extra,
        ]
        low_dir = self.low_dir.resolve()
        actual_open = Image.open

        def only_low_open(path, *args, **kwargs):
            if Path(path).resolve().parent != low_dir:
                raise PermissionError("reference open forbidden")
            return actual_open(path, *args, **kwargs)

        previous = Path.cwd()
        with contextlib.ExitStack() as stack:
            for target, name, value in [
                (runner, "EXPECTED_COUNT", 3),
                (runner, "CHECKPOINT_SHA256", runner.sha256_file(self.receipt)),
                (runner, "load_promptir", lambda path: ToyPromptIR()),
                (runner, "schedule_depthwise", lambda *a, **k: []),
                (runner, "schedule_feedforward", lambda *a, **k: []),
                (runner, "schedule_skip_spill", lambda *a, **k: None),
                (self.tta, "SRTTA", FakeAdapted),
                (self.rddm_net, "ResidualDiffusionModel", lambda *a, **k: object()),
                (torch.nn.Module, "cuda", lambda module, *a, **k: module),
                (torch.Tensor, "cuda", lambda tensor, *a, **k: tensor),
                (torch.cuda, "synchronize", lambda *a, **k: None),
                (torch.cuda, "reset_peak_memory_stats", lambda *a, **k: None),
                (torch.cuda, "max_memory_reserved", lambda *a, **k: 0),
                (torch.cuda, "memory_allocated", lambda *a, **k: 0),
                (torch.cuda, "memory_reserved", lambda *a, **k: 0),
                (torch.cuda, "empty_cache", lambda *a, **k: None),
                (torch.cuda, "get_device_name", lambda *a, **k: "cpu-test"),
                (Image, "open", only_low_open),
                (sys, "argv", argv),
            ]:
                stack.enter_context(patch.object(target, name, value))
            os.chdir(self.work)
            try:
                EVENTS.clear()
                runner.main()
            finally:
                os.chdir(previous)
        return self.root / out

    def test_domain_level_order_and_adapted_inference(self):
        order_out = self.run_main("order", "--order-only")
        self.assertEqual(EVENTS, [])
        order_path = order_out / "adaptation_order.json"
        order = json.loads(order_path.read_bytes())
        out = self.run_main("full", "--expected-order", str(order_path))
        kinds = [kind for kind, _ in EVENTS]
        self.assertEqual(kinds, ["fisher"] * 3 + ["adapt"] * 3 + ["infer"] * 3)
        manifest = json.loads((out / "output_manifest.json").read_bytes())
        self.assertEqual([name for _, name in EVENTS[:3]], [name for name, _ in order["fisher"]])
        self.assertEqual([name for _, name in EVENTS[3:6]], [name for name, _ in order["adaptation"]])
        self.assertEqual(sorted(name for name, _ in order["adaptation"]), NAMES)
        self.assertEqual(manifest["fisher_order"], order["fisher"])
        self.assertEqual(manifest["adaptation_order"], order["adaptation"])
        self.assertEqual(manifest["expected_order_sha256"], runner.sha256_file(order_path))
        self.assertEqual([row["low_name"] for row in manifest["rows"]], NAMES)
        self.assertTrue(all(set(row) == STATIC_ROW_KEYS for row in manifest["rows"]))
        self.assertEqual((manifest["reference_reads"], manifest["metrics"]), (0, 0))
        self.assertTrue(manifest["promotable"] and manifest["model_state_unchanged"])
        updates = manifest["parameter_updates"]
        self.assertEqual(updates["changed_tensor_names"], ["encoder.bias"])
        self.assertEqual((updates["changed_elements"], updates["trainable_elements"]), (3, 12))
        adapted_state = torch.load(out / "adapted_state.pt")
        self.assertEqual(manifest["adapted_state_sha256"], runner.state_sha256(adapted_state))
        bias = adapted_state["encoder.bias"]
        self.assertTrue(torch.allclose(bias, torch.full((3,), 0.03), atol=1e-6))
        with gzip.open(out / "a" / "output.pt.gz", "rb") as stream:
            output = torch.load(stream)
        low = torch.from_numpy(np.array(Image.open(self.low_dir / "a.png"))).permute(2, 0, 1).float() / 255
        self.assertTrue(torch.allclose(output[0], low + bias.view(3, 1, 1), atol=1e-6))
        self.assertEqual(
            manifest["rows"][0]["output_tensor_sha256"], hashlib.sha256(output.numpy().tobytes()).hexdigest()
        )

        order["adaptation"][0][1] = "0" * 64
        tampered = self.root / "tampered_order.json"
        tampered.write_text(json.dumps(order))
        with self.assertRaisesRegex(AssertionError, "adaptation order drift"):
            self.run_main("tampered", "--expected-order", str(tampered))

    def test_smoke_one_is_nonpromotable(self):
        out = self.run_main("smoke", "--smoke-one")
        manifest = json.loads((out / "output_manifest.json").read_bytes())
        self.assertFalse(manifest["promotable"])
        self.assertTrue(manifest["method"].endswith("-SMOKE-NONFINAL"))
        self.assertEqual(manifest["adaptation_scope"], "smoke_image0_only_nonfinal")
        self.assertEqual([name for name, _ in manifest["adaptation_order"]], ["a.png"])
        self.assertEqual([row["low_name"] for row in manifest["rows"]], ["a.png"])

    def test_gt_open_mutation_fails_closed(self):
        for dataset_class in (runner.LowOnlyPromptTrainDataset, runner.LowOnlyPromptTestDataset):
            original = dataset_class.__getitem__
            gt_dir = self.gt_dir

            def mutant(dataset, index, original=original):
                Image.open(gt_dir / "a.png")
                return original(dataset, index)

            with patch.object(dataset_class, "__getitem__", mutant):
                with self.assertRaisesRegex(PermissionError, "reference open forbidden"):
                    self.run_main(f"mutant_{dataset_class.__name__}", "--smoke-one")

    def test_promotable_run_requires_sealed_order(self):
        with self.assertRaisesRegex(AssertionError, "sealed --expected-order"):
            self.run_main("unsealed_full")


if __name__ == "__main__":
    unittest.main()
