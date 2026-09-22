"""Independent verifier for the T072-I low-only dispatch manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from PIL import Image


PAIR_MANIFEST_SHA256 = "3a2ac8c6a0737e02fb562265fe30cbb18af00e5f3535035f172211a4d2d40fcb"
UPSTREAM_COMMIT = "2349d6f0526aff4c2ad9dbf168d93f928bf844f0"
INPUT_METADATA_SHA256 = "d1c66695507f9e4b8f44aa433de8b9a4f6ab8131dbf765dda836c2dcef4b941f"
METHODS = ("ours", "retinexformer", "snr_aware")
SOURCE_ROOT = "shared/t072i/uhdll/input"
EXECUTORS = {"ours": "frozen:final_ours", "retinexformer": "frozen:retinexformer", "snr_aware": "frozen:snr_aware"}
OUTPUT_SUFFIX = {"ours": ".pt", "retinexformer": ".npy", "snr_aware": ".npy"}
OPTIONS = {"batch_size": 1, "crop": False, "downsample": False, "precision_workaround": False, "reference_access": False, "resize": False, "target_specific_normalization": False, "target_specific_tuning": False, "tiling": False}
EXPECTED_BINDINGS = {
    "ours": {"manifest_sha256": "e7f129d931d27531e5f6a14cd72e8c94734c474c40c3c403959cc8f5764e5ab9", "scientific_source": "aa4d920dff4b5b76751c24266e95ac9696d55d90"},
    "retinexformer": {"accepted_binding_sha256": "a9a61665602618adf20c5fb6b05af22da5906c293876fe27342c05a5da833d00", "upstream_commit": "1e9a0efce4b306b6701b824768370ff26066c32a", "checkpoint_sha256": "539bd16c4da6179e45616329f249c4672951b1045193428e1d042c50d4b65a0b", "config_sha256": "5260d0c65878f6a39712f70948be1936d8583531491d832cb59362fffba894ac", "mode": "default_no_gt_mean", "options": {"GT_mean": False, "self_ensemble": False}},
    "snr_aware": {"accepted_binding_sha256": "03ce8bb7f608051ec5c5fd92b7a3e3baea315a2d3978f4c62cc114d226cee875", "upstream_commit": "1113144c82adc8bcc4a9ec27749ed75f196a4e4d", "checkpoint_sha256": "432d29d370e9f674f1b6763d371b4c24569a86d21f0fd45a5797226274d85781", "config_sha256": "fcb29f50538cfd09ec425c83d7f2072477b24f7c2ab4f23507c3e1b37016b3fb", "parameter_sha256": "11d3d667821719bd51e6e608c7876774b001643a28ed85193054bb45428190d4", "mode": "ttie_native_pad16"},
}
FORBIDDEN = {"gt", "reference", "clean", "normal", "label", "metric", "psnr", "ssim"}
PATH_KEYS = re.compile(r"(path|root|entrypoint|command|executable|config|checkpoint|input|output)", re.I)


class VerificationError(ValueError):
    pass


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_guard(value: str) -> None:
    parts = {part.lower() for part in re.split(r"[/\\]+", value) if part}
    if parts.intersection(FORBIDDEN):
        raise VerificationError(f"forbidden target path: {value}")
    if Path(value).is_absolute() or ".." in Path(value).parts:
        raise VerificationError(f"non-relative execution path: {value}")


def scan_path_fields(value: object, key: str = "") -> None:
    if isinstance(value, dict):
        for child_key, child in value.items():
            scan_path_fields(child, str(child_key))
    elif isinstance(value, list):
        for child in value:
            scan_path_fields(child, key)
    elif isinstance(value, str) and PATH_KEYS.search(key):
        path_guard(value)


def trusted_metadata(metadata_path: Path) -> tuple[dict, list[str]]:
    raw = metadata_path.read_bytes()
    try:
        metadata = json.loads(raw.decode("utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise VerificationError(f"invalid trusted metadata: {exc}") from exc
    canonical_digest = hashlib.sha256(canonical_json(metadata)).hexdigest()
    if canonical_digest != INPUT_METADATA_SHA256:
        raise VerificationError("trusted low metadata hash mismatch")
    files = metadata.get("files")
    if not isinstance(files, list) or len(files) != 150:
        raise VerificationError("trusted low metadata count mismatch")
    names = [row.get("name") for row in files]
    if any(not isinstance(name, str) for name in names) or len(set(names)) != 150 or names != sorted(names):
        raise VerificationError("trusted low metadata ordering mismatch")
    return {row["name"]: row for row in files}, names


def verify(manifest: dict, metadata_path: Path, input_root: Path) -> dict:
    trusted, expected_names = trusted_metadata(metadata_path)
    if manifest.get("schema") != "ttie-uhdll-low-dispatch-v1":
        raise VerificationError("schema mismatch")
    if manifest.get("task") != "T072-I" or manifest.get("dataset") != "UHD-LL":
        raise VerificationError("task identity mismatch")
    source = manifest.get("canonical_source")
    if source != {"upstream_commit": UPSTREAM_COMMIT, "pairs_manifest_sha256": PAIR_MANIFEST_SHA256, "input_metadata_sha256": INPUT_METADATA_SHA256, "input_metadata_rows": 150}:
        raise VerificationError("canonical source mismatch")
    if manifest.get("source_root") != SOURCE_ROOT or manifest.get("methods") != list(METHODS):
        raise VerificationError("source or method ordering mismatch")
    if manifest.get("frozen_bindings") != EXPECTED_BINDINGS:
        raise VerificationError("frozen binding mismatch")
    if manifest.get("options") != OPTIONS:
        raise VerificationError("global options mismatch")
    if manifest.get("accounting") != {"inference_runs": 0, "optimizer_runs": 0, "model_fits": 0, "reference_reads": 0, "metrics": 0}:
        raise VerificationError("nonzero accounting")
    scan_path_fields(manifest)
    lows = manifest.get("lows")
    if not isinstance(lows, list) or len(lows) != 150:
        raise VerificationError("low row count mismatch")
    if [row.get("name") for row in lows] != expected_names:
        raise VerificationError("low cohort/order mismatch")
    low_names = set()
    for row in lows:
        name = row.get("name")
        if name in low_names or name not in trusted:
            raise VerificationError("duplicate or untrusted low input")
        low_names.add(name)
        if row.get("relative_path") != name or row.get("source_file_id") != trusted[name]["id"] or row.get("source_size_bytes") != int(trusted[name]["size"]):
            raise VerificationError(f"low metadata mismatch: {name}")
        path_guard(row["relative_path"])
        path = input_root / row["relative_path"]
        if not path.is_file() or path.stat().st_size != int(row.get("bytes", -1)):
            raise VerificationError(f"low file missing/size mismatch: {name}")
        digest = sha256_file(path)
        if digest != row.get("sha256"):
            raise VerificationError(f"low hash mismatch: {name}")
        if row.get("geometry") != [3840, 2160] or row.get("mode") != "RGB" or row.get("dtype") != "uint8":
            raise VerificationError(f"low geometry/dtype mismatch: {name}")
        with Image.open(path) as image:
            image.load()
            if image.format != "JPEG" or image.mode != "RGB" or [image.width, image.height] != [3840, 2160]:
                raise VerificationError(f"low decode mismatch: {name}")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 450 or manifest.get("job_count") != 450:
        raise VerificationError("job count mismatch")
    expected_jobs = {(method, name) for name in expected_names for method in METHODS}
    seen_jobs = set()
    output_paths = set()
    for job in jobs:
        method, name = job.get("method"), job.get("input_name")
        identity = (method, name)
        if identity in seen_jobs or identity not in expected_jobs:
            raise VerificationError("duplicate, missing, or extra job")
        seen_jobs.add(identity)
        if job.get("job_id") != f"{method}:{name}" or job.get("input_relative_path") != name:
            raise VerificationError("job identity/path mismatch")
        if job.get("entrypoint") != EXECUTORS[method] or job.get("binding") != EXPECTED_BINDINGS[method] or job.get("options") != OPTIONS:
            raise VerificationError(f"job binding/options mismatch: {method}:{name}")
        output = job.get("output_relative_path")
        if not isinstance(output, str) or output in output_paths:
            raise VerificationError("output path collision")
        output_paths.add(output)
        path_guard(output)
        expected_output = f"outputs/{method}/{Path(name).stem}{OUTPUT_SUFFIX[method]}"
        if output != expected_output:
            raise VerificationError("unexpected output path")
    if seen_jobs != expected_jobs or len(output_paths) != 450:
        raise VerificationError("incomplete Cartesian job set")
    core = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    root = hashlib.sha256(canonical_json(core)).hexdigest()
    if root != manifest.get("manifest_sha256"):
        raise VerificationError("manifest root hash mismatch")
    return {"status": "PASS", "classification": "UHDLL_FULL_LOW_ONLY_DISPATCH_SEALED", "lows": 150, "jobs": 450, "manifest_sha256": root, "reference_reads": 0, "metrics": 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify(json.loads(args.manifest.read_text(encoding="utf-8")), args.metadata, args.input_root)
    except VerificationError as exc:
        print(json.dumps({"status": "FAIL", "reason": str(exc)}, sort_keys=True))
        raise SystemExit(1)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
