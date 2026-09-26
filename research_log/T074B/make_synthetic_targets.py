"""Synthetic near-black target layouts for execution smokes only (never real data).

A: SID-like npy layout (uint8 BGR, 1424x2128) -> staged at 960x512.
B: LSRW-like PNG layout at 600x400 -> staged native.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def smooth_dark(rng, h, w, level):
    base = cv2.resize(rng.random((5, 7, 3)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    noise = rng.normal(0, 0.01, (h, w, 3))
    return (np.clip(base * level + noise, 0, 1) * 255).round().astype(np.uint8)


def main(root):
    root = Path(root)
    rng = np.random.default_rng(20260925)
    for folder, count in (("10001", 2), ("10002", 1)):
        for side in ("short_sid2", "long_sid2"):
            (root / "A" / side / folder).mkdir(parents=True)
        for k in range(count):
            np.save(root / "A" / "short_sid2" / folder / f"{folder}_0{k}_0.1s.npy",
                    smooth_dark(rng, 1424, 2128, 0.12))
        np.save(root / "A" / "long_sid2" / folder / f"{folder}_00_10s.npy", smooth_dark(rng, 1424, 2128, 0.9))
    for camera, count in (("Huawei", 2), ("Nikon", 1)):
        for side in ("low", "high"):
            (root / "B" / camera / side).mkdir(parents=True)
        for k in range(count):
            Image.fromarray(smooth_dark(rng, 400, 600, 0.12)).save(root / "B" / camera / "low" / f"{k}.png")
            Image.fromarray(smooth_dark(rng, 400, 600, 0.9)).save(root / "B" / camera / "high" / f"{k}.png")


if __name__ == "__main__":
    main(sys.argv[1])
