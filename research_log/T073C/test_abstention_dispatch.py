"""Focused source-side dispatch tests for the authorized no-active rule."""

import gzip
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from research_log.T063A.common import sha, thash
from research_log.T073C.run_ours_ttt_abstain_batch import run_case
from research_log.T073C.verify_ours_ttt_abstain import verify_row


class AbstentionDispatchTest(unittest.TestCase):
    def test_inactive_reuses_exact_step0_and_does_not_call_ttt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            low = root / "image.JPG"
            low.write_bytes(b"synthetic low bytes")
            source = root / "step0" / "image"
            source.mkdir(parents=True)
            tensor = torch.zeros(1, 3, 2160, 3840)
            with gzip.open(source / "output.pt.gz", "wb", compresslevel=1) as stream:
                torch.save(tensor, stream)
            row = {
                "low_name": low.name,
                "low_sha256": sha(low),
                "execution_manifest_sha256": "synthetic-manifest",
                "active": [False] * 4,
                "shape": [1, 3, 2160, 3840],
                "dtype": "torch.float32",
                "output_tensor_sha256": thash(tensor),
                "output_file_sha256": sha(source / "output.pt.gz"),
                "output_manifest_sha256": "synthetic-step0-manifest",
            }
            def forbidden(*_args, **_kwargs):
                raise AssertionError("TTT was called on inactive input")
            output = root / "image"
            receipt = run_case(
                SimpleNamespace(manifest_sha256="synthetic-manifest"),
                low,
                {"name": low.name, "sha256": sha(low)},
                row,
                root / "step0",
                output,
                forbidden,
            )
            self.assertEqual(receipt["decision_status"], "TTT_ABSTAIN_NO_ACTIVE_GATE")
            self.assertEqual(receipt["decision"]["selected_step"], 0)
            self.assertEqual(receipt["optimizer_updates"], 0)
            self.assertEqual(receipt["output_tensor_sha256"], row["output_tensor_sha256"])
            self.assertEqual(sha(output / "output.pt.gz"), row["output_file_sha256"])
            self.assertEqual(verify_row(receipt, {"name": low.name, "sha256": sha(low)}, row, root)[0], True)

    def test_active_dispatches_unchanged_ttt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            low = root / "image.JPG"
            low.write_bytes(b"synthetic low bytes")
            row = {
                "low_name": low.name,
                "low_sha256": sha(low),
                "execution_manifest_sha256": "synthetic-manifest",
                "active": [False, True, False, False],
            }
            calls = []
            def original(model, path, output_dir, expected_low_sha256):
                calls.append((model, path, expected_low_sha256))
                output_dir.mkdir()
                tensor = torch.zeros(1, 3, 2160, 3840)
                with gzip.open(output_dir / "output.pt.gz", "wb", compresslevel=1) as stream:
                    torch.save(tensor, stream)
                return {
                    "low_name": path.name,
                    "low_sha256": sha(path),
                    "decision": {"selected_step": 22},
                    "selected_state_sha256": "a" * 64,
                    "shape": [1, 3, 2160, 3840],
                    "dtype": "torch.float32",
                    "output_tensor_sha256": thash(tensor),
                    "output_file_sha256": sha(output_dir / "output.pt.gz"),
                }
            model = SimpleNamespace(manifest_sha256="synthetic-manifest")
            receipt = run_case(
                model, low, {"name": low.name, "sha256": sha(low)},
                row, root / "unused", root / "image", original,
            )
            self.assertEqual(len(calls), 1)
            self.assertIs(calls[0][0], model)
            self.assertEqual(receipt["decision_status"], "TTT_EXECUTED")
            self.assertEqual(receipt["decision"]["selected_step"], 22)
            self.assertEqual(receipt["output_tensor_sha256"], thash(torch.zeros(1, 3, 2160, 3840)))
            self.assertEqual(json.loads((root / "image" / "decision.json").read_text()), receipt)
            self.assertEqual(verify_row(receipt, {"name": low.name, "sha256": sha(low)}, row, root)[0], False)


if __name__ == "__main__":
    unittest.main()
