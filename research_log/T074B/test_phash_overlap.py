"""pHash sanity: a darkened, resized copy of a scene matches; unrelated scenes do not."""

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phash_overlap import THRESHOLD, hamming, phash_rgb  # noqa: E402


def scene(seed):
    rng = np.random.default_rng(seed)
    base = cv2.resize(rng.random((6, 9, 3)).astype(np.float32), (600, 400), interpolation=cv2.INTER_CUBIC)
    return np.clip(base, 0, 1)


class PHashTest(unittest.TestCase):
    def test_dark_resized_copy_matches(self):
        normal = scene(1)
        dark = np.round(cv2.resize(normal, (960, 512)) * 0.05 * 255) / 255
        noisy = np.clip(dark + np.random.default_rng(3).normal(0, 0.004, dark.shape), 0, 1)
        self.assertLessEqual(hamming(phash_rgb(normal), phash_rgb(noisy)), THRESHOLD)

    def test_unrelated_scenes_differ(self):
        distances = [hamming(phash_rgb(scene(1)), phash_rgb(scene(s))) for s in range(2, 12)]
        self.assertGreater(min(distances), THRESHOLD)


if __name__ == "__main__":
    unittest.main()
