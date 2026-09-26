"""Low-only perceptual-hash overlap check: staged target lows vs LOL-v2 Real Train (low + normal).

Target side reads only the canonical low cache named in ``low_receipt.json``;
target GT is never touched. LOL-v2 Real Train images are source data.
Each image is exposure-normalised (luma / its 99.5th percentile, clipped) so a
near-black target low and a normal-light LOL image of the same scene hash alike,
then hashed with a 64-bit DCT pHash (32x32 INTER_AREA, 8x8 low band, DC
excluded from the median). Pairs with Hamming distance <= THRESHOLD are flagged
for human review of the low images only.
"""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.fft import dctn

THRESHOLD = 10


def phash_rgb(unit_rgb):
    rgb = np.asarray(unit_rgb, dtype=np.float64)
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    scale = max(float(np.percentile(luma, 99.5)), 1e-6)
    luma = np.clip(luma / scale, 0, 1).astype(np.float32)
    small = cv2.resize(luma, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float64)
    band = dctn(small, norm="ortho")[:8, :8].reshape(-1)
    bits = band > np.median(band[1:])
    return int("".join("1" if b else "0" for b in bits), 2)


def load_unit(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB")).astype(np.float32) / 255.0


def hamming(a, b):
    return bin(a ^ b).count("1")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--low-dir", type=Path, required=True)
    p.add_argument("--low-receipt", type=Path, required=True)
    p.add_argument("--lol-train", type=Path, nargs="+", required=True,
                   help="LOL-v2 Real Train Low and Normal folders")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    receipt = json.loads(a.low_receipt.read_bytes())
    source = []
    for folder in a.lol_train:
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
                source.append((str(path), phash_rgb(load_unit(path))))
    assert source, "no LOL-v2 Real Train images found"
    rows, flagged = [], []
    for item in receipt["files"]:
        path = a.low_dir / item["name"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], item["name"]
        h = phash_rgb(load_unit(path))
        distance, nearest = min((hamming(h, s), name) for name, s in source)
        rows.append({"name": item["name"], "min_hamming": distance, "nearest_lol_train": nearest})
        if distance <= THRESHOLD:
            flagged.append(rows[-1])
    report = {"threshold": THRESHOLD, "lol_train_count": len(source), "target_count": len(rows),
              "flagged_count": len(flagged), "flagged": flagged,
              "min_hamming_hist": np.bincount([r["min_hamming"] for r in rows], minlength=65).tolist(),
              "target_reference_reads": 0, "rows": rows}
    a.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("lol_train_count", "target_count", "flagged_count")}))


if __name__ == "__main__":
    main()
