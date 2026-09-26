"""Synthetic paired-vs-low-only Fisher/adaptation equivalence in upstream TTA."""

import copy
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from low_only_loader import LowOnlyPromptTrainDataset


class ToyRestorer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Conv2d(3, 3, 1)

    def forward(self, low, noise_emb=None):
        return torch.sigmoid(self.encoder(low))


class ToyRedegradation:
    def train(self, low, pseudo, name):
        return low + 0.1 * (pseudo - low)


def source_side_step(dataset, base, tta, options):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model = copy.deepcopy(base)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999))
    adapted = tta.SRTTA(options, model, optimizer)
    random.seed(19)
    np.random.seed(19)
    torch.manual_seed(19)
    adapted.compute_fisher(loader)
    fisher_copy = {
        name: [item.clone() for item in value]
        for name, value in adapted.fishers.items()
    }
    names, degraded, _unused_clean = next(iter(loader))
    random.seed(29)
    np.random.seed(29)
    torch.manual_seed(29)
    restored = adapted(loader, degraded, names, ToyRedegradation())
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    return degraded.clone(), fisher_copy, restored.detach().clone(), state


class SourceEquivalenceTest(unittest.TestCase):
    def test_fisher_and_update_ignore_paired_gt(self):
        source = Path(os.environ["DCTTA_SOURCE_ROOT"])
        sys.path.insert(0, str(source))
        import tta
        from utils.dataset_utils import PromptTrainDataset_Simple

        options = SimpleNamespace(iterations=1, teacher_weight=5, compute_fisher=1, fisher_ratio=0.6)
        torch.manual_seed(7)
        base = ToyRestorer()
        original_dir = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            low_dir, gt_dir = root / "low", root / "gt"
            low_dir.mkdir()
            gt_dir.mkdir()
            image = np.uint8(np.arange(32 * 32 * 3).reshape(32, 32, 3) % 256)
            Image.fromarray(image).save(low_dir / "sample.png")
            Image.fromarray(255 - image).save(gt_dir / "sample.png")
            paired = PromptTrainDataset_Simple(str(low_dir), str(gt_dir), patch_size=16)
            low_only = LowOnlyPromptTrainDataset(str(low_dir), patch_size=16)

            def simple_loss(self, prediction, target, *_):
                return (prediction - target).abs().mean()

            try:
                os.chdir(source / "pretrain")  # upstream WaveletTransform fallback uses ./wavelet.mat
                with patch.object(torch.Tensor, "cuda", lambda self: self), \
                        patch.object(tta.SRTTA, "compute_loss2", simple_loss):
                    original = source_side_step(paired, base, tta, options)
                    withheld = source_side_step(low_only, base, tta, options)
            finally:
                os.chdir(original_dir)

            self.assertTrue(torch.equal(original[0], withheld[0]))
            self.assertEqual(original[1].keys(), withheld[1].keys())
            for key in original[1]:
                for source_item, low_item in zip(original[1][key], withheld[1][key]):
                    self.assertTrue(torch.equal(source_item, low_item))
            self.assertTrue(torch.equal(original[2], withheld[2]))
            for key in original[3]:
                self.assertTrue(torch.equal(original[3][key], withheld[3][key]))


if __name__ == "__main__":
    unittest.main()
