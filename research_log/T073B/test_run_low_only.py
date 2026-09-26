"""Synthetic orchestration and GT-access mutation tests for the runner."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import run_low_only


class ToyPromptIR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Conv2d(3, 3, 1)

    def forward(self, low):
        return self.encoder(low)


class FakeAdapted:
    calls = []

    def __init__(self, options, model, optimizer, logger):
        self.model = model
        self.options = options

    def compute_fisher(self, loader):
        name, degraded, surrogate = next(iter(loader))
        assert torch.equal(degraded, surrogate)
        self.calls.append(("fisher", name[0], tuple(degraded.shape)))

    def __call__(self, loader, degraded, names, degeneration_model):
        self.calls.append(("adapt", names[0], tuple(degraded.shape)))
        return self.model(degraded)


class RunnerBoundaryTest(unittest.TestCase):
    def test_low_only_run_and_gt_read_mutation(self):
        source = os.environ["DCTTA_SOURCE_ROOT"]
        sys.path.insert(0, source)
        import tta
        from RDDM import net as rddm_net
        from utils import image_io

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            low_dir = root / "low"
            gt_dir = root / "gt"
            low_dir.mkdir()
            gt_dir.mkdir()
            image = np.uint8(np.arange(352 * 352 * 3).reshape(352, 352, 3) % 256)
            Image.fromarray(image).save(low_dir / "sample.png")
            Image.fromarray(255 - image).save(gt_dir / "sample.png")
            arguments = SimpleNamespace(
                source_root=Path(source), checkpoint=root / "unused.ckpt",
                low_dir=low_dir, output_dir=root / "outputs", mode="dctta",
                smoke_one=True, num_workers=0,
            )
            actual_open = Image.open
            opens = []
            saved = []
            FakeAdapted.calls = []

            def only_low_open(path, *args, **kwargs):
                candidate = Path(path).resolve()
                if candidate.parent != low_dir.resolve():
                    raise PermissionError("reference open forbidden")
                opens.append(candidate)
                return actual_open(path, *args, **kwargs)

            with patch.object(run_low_only, "load_promptir", return_value=ToyPromptIR()), \
                    patch.object(torch.nn.Module, "cuda", lambda self: self), \
                    patch.object(torch.Tensor, "cuda", lambda self: self), \
                    patch.object(tta, "SRTTA", FakeAdapted), \
                    patch.object(rddm_net, "ResidualDiffusionModel", return_value=object()), \
                    patch.object(image_io, "save_image_tensor", side_effect=lambda image, path: saved.append((image.shape, path))), \
                    patch.object(Image, "open", side_effect=only_low_open):
                run_low_only.run(arguments)
                self.assertEqual(opens, [low_dir / "sample.png"] * 3)
                self.assertEqual(FakeAdapted.calls, [
                    ("fisher", "sample.png", (1, 3, 320, 320)),
                    ("adapt", "sample.png", (1, 3, 320, 320)),
                ])
                self.assertEqual(saved[0][0], (1, 3, 352, 352))
                self.assertEqual(Path(saved[0][1]).name, "sample.png")

                original_getitem = run_low_only.LowOnlyPromptTrainDataset.__getitem__

                def mutant_getitem(dataset, index):
                    Image.open(gt_dir / "sample.png")
                    return original_getitem(dataset, index)

                with patch.object(run_low_only.LowOnlyPromptTrainDataset, "__getitem__", mutant_getitem):
                    with self.assertRaisesRegex(PermissionError, "reference open forbidden"):
                        run_low_only.run(arguments)


if __name__ == "__main__":
    unittest.main()
