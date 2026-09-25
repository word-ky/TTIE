"""T074-B dataset-generic low-only target staging.

Reads ONLY the low files of the declared test list and writes:

* ``low/``: canonical lossless low cache (8-bit RGB PNG when the source is
  uint8, otherwise float32 RGB ``.npy`` in [0, 1]); nothing else in the folder;
* ``low_receipt.json``: the only file runners receive (names, hashes, H x W,
  cluster, sealed darkness statistic);
* ``reference_opaque_manifest.json``: GT paths plus raw-byte SHA256, hashed
  with ``os.read`` only. GT payloads are never decoded; every image decoder is
  guarded against GT paths for the whole process.

Test-list rules follow the official SNR-Aware / Retinexformer loaders (see
``research_log/T074B/protocol_draft.md``). Any missing pair, duplicate, name
mismatch or unexpected format fails closed.
"""

import argparse
import contextlib
import glob
import hashlib
import json
import os
import os.path as osp
from pathlib import Path

import numpy as np

# Must stay True until the phase-3 reference gate. stage() refuses to run otherwise.
REFERENCE_DECODE_FORBIDDEN = True

SNR_SIZE_WH = (960, 512)  # SNR-Aware / Retinexformer train_size for SID/SMID/SDSD test
SDSD_TEST_DIRS = {
    "sdsd_in": "pair11,pair21,pair1,pair19,pair4,pair9".split(","),
    "sdsd_out": ("MVI_0898,MVI_0928,MVI_0906,MVI_0975,MVI_1001,"
                 "MVI_0997,MVI_1003,MVI_1026,MVI_1030,MVI_1032").split(","),
}
DARKNESS = {"median_max": 0.10, "per_image_max": 0.15, "fraction_min": 0.80}
LUMA = (0.299, 0.587, 0.114)


class ReferenceAccessError(PermissionError):
    pass


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def opaque_sha256(path):
    """Raw-byte hash of a reference file via os.read; never a decoder."""
    digest = hashlib.sha256()
    size = 0
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    try:
        while True:
            block = os.read(fd, 8 << 20)
            if not block:
                break
            digest.update(block)
            size += len(block)
    finally:
        os.close(fd)
    return digest.hexdigest(), size


def official_glob(root):
    """``util.glob_file_list``: sorted(glob(root/*))."""
    return sorted(glob.glob(osp.join(str(root), "*")))


# ---------------------------------------------------------------- guard


class DecodeGuard:
    """Refuses every image/array decode of a path under a reference root or in the GT set."""

    def __init__(self, roots, files=()):
        self.roots = [osp.realpath(str(r)) for r in roots]
        self.files = {osp.realpath(str(f)) for f in files}

    def forbidden(self, path):
        if not isinstance(path, (str, bytes, os.PathLike)):
            return False
        real = osp.realpath(os.fsdecode(path))
        if real in self.files:
            return True
        return any(real == root or real.startswith(root + os.sep) for root in self.roots)

    def check(self, path):
        if self.forbidden(path):
            raise ReferenceAccessError(f"reference decode forbidden before the reference gate: {path}")

    @contextlib.contextmanager
    def active(self):
        import cv2
        import PIL.Image

        originals = []

        def wrap(owner, name):
            original = getattr(owner, name)

            def guarded(target, *args, **kwargs):
                self.check(target)
                return original(target, *args, **kwargs)

            originals.append((owner, name, original))
            setattr(owner, name, guarded)

        wrap(np, "load")
        wrap(np, "fromfile")
        wrap(cv2, "imread")
        wrap(PIL.Image, "open")
        try:
            yield self
        finally:
            for owner, name, original in reversed(originals):
                setattr(owner, name, original)


# ---------------------------------------------------------------- test lists


def _item(name, cluster, low, gt, **extra):
    return dict(name=name, cluster=cluster, low_path=str(low), gt_path=str(gt), **extra)


def _paired_folders(lq_root, gt_root, keep):
    """Official positional zip of sorted folders, required to agree by basename."""
    lq, gt = official_glob(lq_root), official_glob(gt_root)
    pairs = []
    for index, lq_folder in enumerate(lq):
        name = osp.basename(lq_folder)
        if not keep(name):
            continue
        assert index < len(gt), f"no GT folder at position {index} for {name}"
        assert osp.basename(gt[index]) == name, (
            f"official positional pairing mismatch: {name} vs {osp.basename(gt[index])}")
        assert osp.isdir(lq_folder) and osp.isdir(gt[index]), name
        pairs.append((name, lq_folder, gt[index]))
    return pairs


def list_sid(root):
    """Dataset_SIDImage (phase test): folders whose name starts with '1'; GT = sorted(long)[0]."""
    root = Path(root)
    items = []
    for name, lq_folder, gt_folder in _paired_folders(
            root / "short_sid2", root / "long_sid2", lambda n: n[:1] == "1"):
        lows, gts = official_glob(lq_folder), official_glob(gt_folder)
        assert lows, f"empty low folder {name}"
        assert gts, f"missing GT for {name}"
        for low in lows:
            items.append(_item(f"{name}__{Path(low).stem}.png", name, low, gts[0],
                               gt_folder_file_count=len(gts)))
    return items


def list_smid(root, test_list, frames):
    """Dataset_SMIDImage / SNR SMID test: listed sequences; GT = first non-ARW/non-half file."""
    root = Path(root)
    testing = [line.strip() for line in Path(test_list).read_text().splitlines()]
    testing = [name for name in testing if name]
    assert len(testing) == len(set(testing)), "duplicate sequence in test list"
    pairs = _paired_folders(root / "SMID_LQ_np", root / "SMID_Long_np", lambda n: n in testing)
    assert sorted(p[0] for p in pairs) == sorted(testing), "test-list sequence missing on disk"
    items = []
    for name, lq_folder, gt_folder in pairs:
        lows = official_glob(lq_folder)
        if frames:
            lows = lows[:frames]
        gts = [p for p in official_glob(gt_folder) if ".ARW" not in p and "half" not in p]
        assert lows, f"empty low folder {name}"
        assert gts, f"missing GT for {name}"
        for low in lows:
            items.append(_item(f"{name}__{Path(low).stem}.png", name, low, gts[0],
                               gt_folder_file_count=len(gts)))
    return items


def list_sdsd(root, subset, frames):
    """Dataset_SDSDImage / SNR SDSD test: testing_dir or name.split('_2')[0]; per-frame GT by index."""
    testing = SDSD_TEST_DIRS[subset]
    root = Path(root)
    keep = lambda n: n in testing or n.split("_2")[0] in testing  # noqa: E731
    items = []
    pairs = _paired_folders(root / "input", root / "GT", keep)
    assert {n.split("_2")[0] for n, _, _ in pairs} >= set(testing), "test video missing on disk"
    for name, lq_folder, gt_folder in pairs:
        lows, gts = official_glob(lq_folder), official_glob(gt_folder)
        if frames:
            lows, gts = lows[:frames], gts[:frames]
        assert lows and len(lows) == len(gts), f"Different number of images in LQ and GT folders: {name}"
        # Official rule pairs LQ[idx] with GT[idx] positionally after sorted(glob(...)) (dataset_SDSD_test.py
        # [idx:idx+1]); the two cameras' own frame counters need not agree (verified: on the real SDSD_indoor
        # mirror, every pair has one constant per-video LQ-GT filename-number offset, e.g. pair1 = +2,
        # pair11 = -54), so basename equality is not part of the official rule and is not required here.
        for low, gt in zip(lows, gts):
            items.append(_item(f"{name}__{Path(low).stem}.png", name.split("_2")[0], low, gt))
    return items


def list_lsrw(root):
    """LSRW Eval: <root>/{Huawei,Nikon}/{low,high}, paired by identical filename."""
    root = Path(root)
    items = []
    for camera in ("Huawei", "Nikon"):
        low_dir, high_dir = root / camera / "low", root / camera / "high"
        lows = sorted(p.name for p in low_dir.iterdir())
        highs = sorted(p.name for p in high_dir.iterdir())
        assert lows, f"empty {low_dir}"
        assert lows == highs, f"LSRW {camera} low/high filename sets differ"
        for name in lows:
            items.append(_item(f"{camera}__{Path(name).stem}.png", f"{camera}__{Path(name).stem}",
                               low_dir / name, high_dir / name))
    return items


def reference_roots(dataset, root):
    root = Path(root)
    return {
        "sid": [root / "long_sid2"],
        "smid": [root / "SMID_Long_np"],
        "sdsd_in": [root / "GT"],
        "sdsd_out": [root / "GT"],
        "lsrw": [root / "Huawei" / "high", root / "Nikon" / "high"],
    }[dataset]


def build_list(args):
    if args.dataset == "sid":
        return list_sid(args.root)
    if args.dataset == "smid":
        assert args.smid_test_list, "--smid-test-list is required"
        return list_smid(args.root, args.smid_test_list, args.frames_per_sequence)
    if args.dataset in SDSD_TEST_DIRS:
        return list_sdsd(args.root, args.dataset, args.frames_per_sequence)
    return list_lsrw(args.root)


# ---------------------------------------------------------------- low decode


def decode_low(path, geometry, guard):
    """Canonical low as the official loader delivers it, before /255: (RGB array, facts)."""
    guard.check(path)
    if str(path).endswith(".npy"):
        array = np.load(path, allow_pickle=False)
        facts = {"source_dtype": str(array.dtype), "source_shape": list(array.shape)}
        assert array.ndim == 3 and array.shape[2] == 3, f"unsupported npy shape {array.shape}: {path}"
        if geometry == "snr960x512":
            import cv2
            array = cv2.resize(array, SNR_SIZE_WH)  # util.read_img2, default INTER_LINEAR
        return np.ascontiguousarray(array[:, :, ::-1]), facts  # read_img_seq2 [2, 1, 0]
    assert geometry == "native", "image-file sources are staged at native geometry"
    from PIL import Image

    with Image.open(path) as image:
        orientation = image.getexif().get(274)
        assert orientation in (None, 1), f"EXIF orientation {orientation}: {path}"
        assert image.mode == "RGB", f"mode {image.mode}: {path}"
        array = np.asarray(image).copy()
        facts = {"source_dtype": str(array.dtype), "source_shape": list(array.shape),
                 "source_format": image.format}
    return array, facts


def to_unit(array):
    return array.astype(np.float32) / np.float32(255.0)  # read_img2: astype(float32) / 255.


def luma_mean(unit_rgb):
    rgb = unit_rgb.astype(np.float64)
    return float((LUMA[0] * rgb[..., 0] + LUMA[1] * rgb[..., 1] + LUMA[2] * rgb[..., 2]).mean())


def darkness_summary(values):
    values = np.asarray(values, dtype=np.float64)
    assert values.size and np.isfinite(values).all()
    fraction = float((values <= DARKNESS["per_image_max"]).mean())
    median = float(np.median(values))
    return {
        "statistic": "per-image mean of 0.299R+0.587G+0.114B on canonical low RGB in [0,1]",
        "count": int(values.size),
        "median": median,
        "p80": float(np.percentile(values, 80)),
        "p90": float(np.percentile(values, 90)),
        "fraction_le_0.15": fraction,
        "criterion": DARKNESS,
        "pass": bool(median <= DARKNESS["median_max"] and fraction >= DARKNESS["fraction_min"]),
    }


def write_cache(rgb, path):
    if rgb.dtype == np.uint8:
        from PIL import Image

        Image.fromarray(rgb, "RGB").save(path, format="PNG")
        with Image.open(path) as check:
            assert np.array_equal(np.asarray(check.convert("RGB")), rgb), "PNG roundtrip mismatch"
        return path, "png-rgb-uint8"
    path = path.with_suffix(".npy")
    unit = to_unit(rgb)
    np.save(path, unit)
    assert np.array_equal(np.load(path), unit)
    return path, "npy-rgb-float32-unit"


# ---------------------------------------------------------------- stage


def stage(args):
    if not REFERENCE_DECODE_FORBIDDEN:
        raise ReferenceAccessError("REFERENCE_DECODE_FORBIDDEN must be True before the reference gate")
    items = build_list(args)
    assert items, "empty test list"
    items.sort(key=lambda item: item["name"])
    names = [item["name"] for item in items]
    assert len(set(names)) == len(names), "duplicate canonical names"
    assert len({Path(n).stem for n in names}) == len(names), "duplicate canonical stems"
    if args.expected_count is not None:
        assert len(items) == args.expected_count, f"{len(items)} items, declared {args.expected_count}"
    low_paths = {osp.realpath(item["low_path"]) for item in items}
    gt_paths = {osp.realpath(item["gt_path"]) for item in items}
    assert len(low_paths) == len(items), "a low file appears twice"
    assert not (low_paths & gt_paths), "a file is both low and GT"
    for item in items:
        assert osp.isfile(item["low_path"]), f"missing low {item['low_path']}"
        assert osp.isfile(item["gt_path"]), f"missing GT {item['gt_path']}"
    guard = DecodeGuard(reference_roots(args.dataset, args.root), gt_paths)
    for low in low_paths:
        assert not guard.forbidden(low), f"low file under a reference root: {low}"
    out = Path(args.out)
    if args.list_only:
        out.mkdir(parents=True, exist_ok=False)
        listing = [{k: item[k] for k in ("name", "cluster", "low_path")} for item in items]
        (out / "test_list.json").write_text(json.dumps(
            {"dataset": args.dataset, "count": len(items), "clusters": len({i["cluster"] for i in items}),
             "items": listing}, indent=2))
        return None
    (out / "low").mkdir(parents=True, exist_ok=False)
    root = Path(args.root)
    rows, pairs = [], []
    with guard.active():
        for item in items:
            rgb, facts = decode_low(item["low_path"], args.geometry, guard)
            cache, cache_format = write_cache(rgb, out / "low" / item["name"])
            row = {
                "name": cache.name,
                "sha256": sha256_file(cache),
                "height": int(rgb.shape[0]),
                "width": int(rgb.shape[1]),
                "cluster": item["cluster"],
                "cache_format": cache_format,
                "source_relpath": osp.relpath(item["low_path"], root),
                "source_sha256": sha256_file(item["low_path"]),
                **facts,
                "luma_mean": luma_mean(to_unit(rgb)),
            }
            rows.append(row)
        for item, row in zip(items, rows):
            gt_hash, gt_bytes = opaque_sha256(item["gt_path"])
            pairs.append({"name": row["name"], "cluster": item["cluster"],
                          "gt_relpath": osp.relpath(item["gt_path"], root),
                          "gt_raw_sha256": gt_hash, "gt_bytes": gt_bytes,
                          **({"gt_folder_file_count": item["gt_folder_file_count"]}
                             if "gt_folder_file_count" in item else {})})
    source_hashes = [row["source_sha256"] for row in rows]
    assert len(set(source_hashes)) == len(source_hashes), "duplicate low source bytes"
    assert not (set(source_hashes) & {p["gt_raw_sha256"] for p in pairs}), "a low equals a GT byte-for-byte"
    owner = {}
    for pair in pairs:
        previous = owner.setdefault(pair["gt_raw_sha256"], pair["cluster"])
        assert previous == pair["cluster"], f"identical GT bytes across clusters {previous} / {pair['cluster']}"
    present = sorted(os.listdir(out / "low"))
    assert present == sorted(row["name"] for row in rows), "cache folder must hold exactly the receipt files"
    rule = {
        "dataset": args.dataset,
        "root": str(root),
        "geometry": args.geometry,
        "frames_per_sequence": args.frames_per_sequence,
        "smid_test_list_sha256": sha256_file(args.smid_test_list) if args.smid_test_list else None,
        "sdsd_testing_dir": SDSD_TEST_DIRS.get(args.dataset),
        "order": "sorted canonical name",
        "stage_script_sha256": sha256_file(__file__),
    }
    receipt = {
        "task": "T074-B-stage",
        "rule": rule,
        "count": len(rows),
        "cluster_count": len({row["cluster"] for row in rows}),
        "reference_reads": 0,
        "reference_decodes": 0,
        "metrics": 0,
        "darkness": darkness_summary([row["luma_mean"] for row in rows]),
        "files": rows,
    }
    receipt_path = out / "low_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))
    opaque = {
        "task": "T074-B-reference-opaque",
        "rule": rule,
        "low_receipt_sha256": sha256_file(receipt_path),
        "count": len(pairs),
        "unique_gt_count": len(owner),
        "reference_decodes": 0,
        "gt_access": "raw-byte SHA256 via os.read only; decoders guarded",
        "pairs": pairs,
    }
    (out / "reference_opaque_manifest.json").write_text(json.dumps(opaque, indent=2))
    return receipt


def parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", required=True, choices=["sid", "smid", "sdsd_in", "sdsd_out", "lsrw"])
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--geometry", choices=["snr960x512", "native"], required=True)
    p.add_argument("--frames-per-sequence", type=int, default=0, help="0 = all frames")
    p.add_argument("--smid-test-list", type=Path)
    p.add_argument("--expected-count", type=int)
    p.add_argument("--list-only", action="store_true", help="enumerate paths only; no decode, no hashing")
    return p


def main():
    receipt = stage(parser().parse_args())
    if receipt is not None:
        print(json.dumps({k: receipt[k] for k in ("count", "cluster_count", "darkness")}, indent=2))


if __name__ == "__main__":
    main()
