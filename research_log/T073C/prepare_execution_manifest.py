"""Rebind frozen T070-A assets/environment to the paid-card execution host."""

import argparse
import json
from pathlib import Path

from research_log.T063A.common import sha, write
from research_log.T070A.manifest import environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    assert sha(args.frozen) == "e7f129d931d27531e5f6a14cd72e8c94734c474c40c3c403959cc8f5764e5ab9"
    manifest = json.loads(args.frozen.read_bytes())
    manifest["assets"]["clip"]["path"] = str(args.asset_root / "open_clip_pytorch_model.bin")
    manifest["assets"]["prototypes"]["path"] = str(args.asset_root / "prototypes.pt")
    manifest["assets"]["gate"]["path"] = str(args.asset_root / "T022A_gate.json")
    for item in manifest["assets"].values():
        assert sha(item["path"]) == item["sha256"], item["path"]
    for path, expected in manifest["source_binding"].items():
        assert sha(path) == expected, path
    manifest["environment"] = environment()
    manifest["execution_origin"] = {
        "frozen_manifest_sha256": sha(args.frozen),
        "change": "asset paths and execution environment only; source, constants and asset bytes unchanged",
    }
    write(args.out, manifest)
    print(sha(args.out))


if __name__ == "__main__":
    main()
