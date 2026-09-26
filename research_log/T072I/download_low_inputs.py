"""Download only the canonical UHD-LL low-input files listed in input_metadata.json."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import requests


def download(metadata_path: Path, output_root: Path) -> dict:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    files = metadata.get("files")
    if not isinstance(files, list) or len(files) != 150:
        raise ValueError("expected exactly 150 low-input metadata rows")
    if any("gt" in row or "reference" in row for row in files):
        raise ValueError("low-input metadata unexpectedly contains reference fields")
    output_root.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "TTIE-T072I-low-dispatch/1.0"})
    rows = []
    for row in sorted(files, key=lambda item: item["name"]):
        name = row["name"]
        if Path(name).name != name or Path(name).suffix.lower() != ".jpg":
            raise ValueError(f"unsafe low filename: {name!r}")
        target = output_root / name
        url = (
            "https://drive.usercontent.google.com/download?id="
            f"{row['id']}&export=download&confirm=t"
        )
        response = session.get(url, timeout=60, stream=True)
        response.raise_for_status()
        part = target.with_suffix(target.suffix + ".part")
        digest = hashlib.sha256()
        total = 0
        with part.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    digest.update(chunk)
                    total += len(chunk)
        if total != int(row["size"]):
            part.unlink(missing_ok=True)
            raise ValueError(f"size mismatch for {name}: {total} != {row['size']}")
        part.replace(target)
        rows.append({"name": name, "bytes": total, "sha256": digest.hexdigest()})
    return {"count": len(rows), "files": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = download(args.metadata, args.output_root)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
