"""Create one deterministic low-only image for execution-path smoke tests."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    y, x = np.indices((352, 352), dtype=np.uint32)
    image = np.stack((x % 256, y % 256, (x + 2 * y) % 256), axis=-1).astype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(args.output)


if __name__ == "__main__":
    main()
