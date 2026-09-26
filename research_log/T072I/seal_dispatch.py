"""Build the immutable low-only UHD-LL dispatch manifest for T072-I."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from PIL import Image

from research_log.T072E.constants import EXPECTED_BINDINGS


TASK = "T072-I"
PAIR_MANIFEST_SHA256 = "3a2ac8c6a0737e02fb562265fe30cbb18af00e5f3535035f172211a4d2d40fcb"
UPSTREAM_COMMIT = "2349d6f0526aff4c2ad9dbf168d93f928bf844f0"
METHODS = ("ours", "retinexformer", "snr_aware")
SOURCE_ROOT = "shared/t072i/uhdll/input"
EXECUTORS = {
    "ours": "frozen:final_ours",
    "retinexformer": "frozen:retinexformer",
    "snr_aware": "frozen:snr_aware",
}
OUTPUT_SUFFIX = {"ours": ".pt", "retinexformer": ".npy", "snr_aware": ".npy"}
OPTIONS = {
    "batch_size": 1,
    "crop": False,
    "downsample": False,
    "precision_workaround": False,
    "reference_access": False,
    "resize": False,
    "target_specific_normalization": False,
    "target_specific_tuning": False,
    "tiling": False,
}
FORBIDDEN_PATH_PARTS = {"gt", "reference", "clean", "normal", "label", "metric", "psnr", "ssim"}


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def guard_relative(name: str) -> None:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or relative.name != name:
        raise ValueError(f"unsafe relative path: {name!r}")
    lowered = {part.lower() for part in relative.parts}
    if lowered.intersection(FORBIDDEN_PATH_PARTS):
        raise PermissionError(f"forbidden target path: {name!r}")


def metadata_digest(metadata: dict) -> str:
    return hashlib.sha256(canonical_json(metadata)).hexdigest()


def probe_rows(metadata_path: Path, input_root: Path) -> tuple[dict, list[dict]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    files = metadata.get("files")
    if not isinstance(files, list) or len(files) != 150:
        raise ValueError("canonical low metadata must contain 150 rows")
    names = [row.get("name") for row in files]
    if any(not isinstance(name, str) for name in names) or len(set(names)) != 150:
        raise ValueError("canonical low metadata has duplicate or invalid names")
    rows = []
    for source in sorted(files, key=lambda item: item["name"]):
        name = source["name"]
        guard_relative(name)
        path = input_root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(source["size"]):
            raise ValueError(f"size mismatch for {name}")
        with Image.open(path) as image:
            image.load()
            if image.format != "JPEG" or image.mode != "RGB":
                raise ValueError(f"unexpected image encoding for {name}")
            geometry = [image.width, image.height]
            mode = image.mode
        if geometry != [3840, 2160]:
            raise ValueError(f"native geometry mismatch for {name}: {geometry}")
        rows.append(
            {
                "name": name,
                "relative_path": name,
                "source_file_id": source["id"],
                "source_size_bytes": int(source["size"]),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "geometry": geometry,
                "mode": mode,
                "dtype": "uint8",
            }
        )
    trusted = {"metadata": metadata, "metadata_sha256": metadata_digest(metadata)}
    return trusted, rows


def build_manifest(metadata_path: Path, input_root: Path) -> dict:
    trusted, rows = probe_rows(metadata_path, input_root)
    jobs = []
    for row in rows:
        stem = Path(row["name"]).stem
        for method in METHODS:
            jobs.append(
                {
                    "job_id": f"{method}:{row['name']}",
                    "method": method,
                    "input_name": row["name"],
                    "input_relative_path": row["relative_path"],
                    "output_relative_path": f"outputs/{method}/{stem}{OUTPUT_SUFFIX[method]}",
                    "entrypoint": EXECUTORS[method],
                    "binding": copy.deepcopy(EXPECTED_BINDINGS[method]),
                    "options": copy.deepcopy(OPTIONS),
                }
            )
    manifest = {
        "schema": "ttie-uhdll-low-dispatch-v1",
        "task": TASK,
        "dataset": "UHD-LL",
        "canonical_source": {
            "upstream_commit": UPSTREAM_COMMIT,
            "pairs_manifest_sha256": PAIR_MANIFEST_SHA256,
            "input_metadata_sha256": trusted["metadata_sha256"],
            "input_metadata_rows": 150,
        },
        "source_root": SOURCE_ROOT,
        "methods": list(METHODS),
        "frozen_bindings": copy.deepcopy(EXPECTED_BINDINGS),
        "lows": rows,
        "jobs": jobs,
        "job_count": len(jobs),
        "read_scope": {
            "allowed_payload": "degraded_input_only",
            "forbidden_payloads": ["gt", "reference", "clean", "normal-light"],
        },
        "options": copy.deepcopy(OPTIONS),
        "accounting": {
            "inference_runs": 0,
            "optimizer_runs": 0,
            "model_fits": 0,
            "reference_reads": 0,
            "metrics": 0,
        },
    }
    manifest["manifest_sha256"] = hashlib.sha256(canonical_json(manifest)).hexdigest()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.metadata, args.input_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest_sha256": manifest["manifest_sha256"], "lows": len(manifest["lows"]), "jobs": len(manifest["jobs"])}, sort_keys=True))


if __name__ == "__main__":
    main()
