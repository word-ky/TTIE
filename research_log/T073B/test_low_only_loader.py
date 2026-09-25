"""Source-side DCTTA dataset equivalence; synthetic images only."""

import os
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from low_only_loader import LowOnlyPromptTrainDataset


class LowOnlyLoaderTest(unittest.TestCase):
    def test_degraded_sequence_matches_official_and_never_opens_gt(self):
        source = os.environ["DCTTA_SOURCE_ROOT"]
        sys.path.insert(0, source)
        from utils.dataset_utils import PromptTrainDataset_Simple

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            low_dir = root / "low"
            gt_dir = root / "gt"
            low_dir.mkdir()
            gt_dir.mkdir()
            for index in range(3):
                values = np.arange(48 * 52 * 3, dtype=np.uint32).reshape(48, 52, 3)
                image = np.uint8((values + index * 13) % 256)
                Image.fromarray(image).save(low_dir / f"{index}.png")
                Image.fromarray(255 - image).save(gt_dir / f"{index}.png")

            official = PromptTrainDataset_Simple(str(low_dir), str(gt_dir), patch_size=16)
            random.seed(311)
            expected = [official[index][1] for index in range(3)]

            low_only = LowOnlyPromptTrainDataset(str(low_dir), patch_size=16)
            opened = []
            original_open = Image.open

            def record_open(path, *args, **kwargs):
                opened.append(Path(path).resolve())
                return original_open(path, *args, **kwargs)

            random.seed(311)
            with patch.object(Image, "open", side_effect=record_open):
                actual = [low_only[index][1] for index in range(3)]

            self.assertEqual(len(low_only), 3)
            self.assertEqual(opened, [(low_dir / f"{index}.png").resolve() for index in range(3)])
            for reference, candidate in zip(expected, actual):
                self.assertTrue(torch.equal(reference, candidate))


if __name__ == "__main__":
    unittest.main()
