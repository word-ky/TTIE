"""Ours generic dispatch tests; run from the frozen Ours source root (needs research_log.T063A/T073C)."""

import gzip
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from research_log.T063A.common import sha, thash
from research_log.T073C.run_ours_ttt_abstain_batch import run_case as frozen_run_case

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ttt_generic", HERE / "run_ours_ttt_abstain_generic.py")
generic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generic)
spec = importlib.util.spec_from_file_location("verify_generic", HERE / "verify_generic_outputs.py")
vg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vg)

TIMING = {"whole_run_seconds", "peak_gpu_memory_bytes"}


def inactive_case(root, h, w):
    low = root / "image.png"
    low.write_bytes(b"synthetic low bytes")
    source = root / "step0" / "image"
    source.mkdir(parents=True)
    tensor = torch.rand(1, 3, h, w)
    with gzip.open(source / "output.pt.gz", "wb", compresslevel=1) as stream:
        torch.save(tensor, stream)
    step0 = {
        "low_name": low.name, "low_sha256": sha(low), "execution_manifest_sha256": "m",
        "active": [False] * 4, "shape": [1, 3, h, w], "dtype": "torch.float32",
        "output_tensor_sha256": thash(tensor), "output_file_sha256": sha(source / "output.pt.gz"),
        "output_manifest_sha256": "step0-manifest",
    }
    expected = {"name": low.name, "sha256": sha(low), "height": h, "width": w}
    return low, step0, expected


def forbidden(*_a, **_k):
    raise AssertionError("TTT called on inactive input")


class GenericOursTest(unittest.TestCase):
    def test_generic_equals_frozen_on_4k(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            low, step0, expected = inactive_case(root, 2160, 3840)
            model = SimpleNamespace(manifest_sha256="m")
            a = frozen_run_case(model, low, expected, step0, root / "step0", root / "frozen" / "image", forbidden)
            b = generic.run_case(model, low, expected, step0, root / "step0", root / "generic" / "image", forbidden)
            self.assertEqual({k: v for k, v in a.items() if k not in TIMING},
                             {k: v for k, v in b.items() if k not in TIMING})

    def test_generic_inactive_512x960_reuses_step0(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            low, step0, expected = inactive_case(root, 512, 960)
            receipt = generic.run_case(SimpleNamespace(manifest_sha256="m"), low, expected, step0,
                                       root / "step0", root / "image", forbidden)
            self.assertEqual(receipt["decision_status"], "TTT_ABSTAIN_NO_ACTIVE_GATE")
            self.assertEqual(receipt["output_file_sha256"], step0["output_file_sha256"])
            self.assertTrue(vg.verify_abstention(receipt, step0))
            vg.verify_row(receipt, expected, root)

    def test_geometry_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            low, step0, expected = inactive_case(root, 512, 960)
            expected["height"] = 400
            with self.assertRaises(AssertionError):
                generic.run_case(SimpleNamespace(manifest_sha256="m"), low, expected, step0,
                                 root / "step0", root / "image", forbidden)

    def test_active_dispatches_ttt(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            low, step0, expected = inactive_case(root, 512, 960)
            step0["active"] = [False, True, False, False]
            calls = []

            def original(model, path, output_dir, expected_low_sha256):
                calls.append(path)
                output_dir.mkdir()
                (output_dir / "decision.json").write_text("{}")
                return {"low_name": path.name}

            receipt = generic.run_case(SimpleNamespace(manifest_sha256="m"), low, expected, step0,
                                       root / "unused", root / "image", original)
            self.assertEqual(len(calls), 1)
            self.assertEqual(receipt["decision_status"], "TTT_EXECUTED")


if __name__ == "__main__":
    unittest.main()
