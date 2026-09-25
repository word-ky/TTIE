"""Dataset-generic low-only MR. Illuminate runner at native input resolution (T073-D).

The model computation is the unmodified official `main.py` at commit 013e013e:
`seed_everything(config.seed)`, `DiffusionPriorEnhancer(config)`,
`find_image_paths`, `DiffusionPriorEnhancer.process` and `save_output_image`,
called in the order used by `main.run_inference` (main.py:588-620). The
wrapper differs from `run_inference` only in execution plumbing:

* it does not catch per-image exceptions (official main.py:614 swallows them);
  every failure stops the run;
* it skips the brightness-scaled *input* baseline (main.py:607-613), which is
  not the method output and consumes no model state or RNG;
* `main.DEFAULT_MODEL_ID` is pointed at a pinned local SD1.5 diffusers snapshot
  with the Hugging Face hub forced offline (asset location only);
* the custom VAE's bound `decode` is wrapped by an observer that returns the
  official result unchanged and records the same pre-quantization float the
  official `_decode_latents` computes (main.py:333-335);
* `PIL.Image.open` accepts only the planned receipt low files and
  `cv2.imread` is denied, so no other image can be decoded.

Frozen metric artifact per image: `output.pt.gz`, a float32 `[1, 3, H, W]`
RGB tensor at the low image's own geometry (PIL size, as the official code
reads it). Normally it equals `float32(v) / 255`, where `v` is the official
pre-quantization array `rearrange(decoded) * 127.5 + 127.5` (unclipped). If a
side is not a multiple of 8, the official code LANCZOS-resizes up to /8 and
maps its uint8 output back with `cv2.resize` (main.py:291-299, 531-532); the
runner then applies that same map to `clip(float32(v) / 255, 0, 1)` and also
keeps the native float. Before saving, the runner asserts bitwise that the
official uint8 recomputed from the captured `v` equals what `process` returned.
The official 8-bit file (its format follows the input extension) and a
lossless PNG of the native official uint8 are kept for provenance only.

No reference/GT path exists in this program; it computes no metric.
"""

import os

# Asset resolution must never reach the network or another cache entry.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["DIFFUSERS_OFFLINE"] = "1"

import argparse
import gzip
import hashlib
import json
import logging
import platform
import random
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PIN_FILE = HERE / "mri_source_pin.json"
BYTES_PER_PIXEL_BUDGET = 30  # float32 metric (+ native when resized) + PNG + official file
OFFICIAL_DEFAULTS = {
    "seed": 1,
    "target_avg": 30.0,
    "n_timesteps": 25,
    "guidance_scale": 1.0,
    "prompt": "",
    "negative_prompt": "",
    "enable_xformers": True,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def self_sha256_lf():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def state_sha256(module):
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(tuple(array.shape)).encode())
        digest.update(array.numpy().tobytes())
    return digest.hexdigest()


def rng_digest():
    """Digest of every global RNG the official code could consume."""
    parts = [torch.get_rng_state().numpy().tobytes()]
    if torch.cuda.is_available():
        parts.extend(state.numpy().tobytes() for state in torch.cuda.get_rng_state_all())
    numpy_state = np.random.get_state()
    parts.append(numpy_state[1].tobytes() + repr(numpy_state[2:]).encode())
    parts.append(repr(random.getstate()).encode())
    return hashlib.sha256(b"".join(parts)).hexdigest()


def verify_source_pin(source_root, pin_path=PIN_FILE):
    pin = json.loads(Path(pin_path).read_bytes())
    mismatches = []
    for rel, expected in sorted(pin["files"].items()):
        path = Path(source_root) / rel
        if not path.is_file():
            mismatches.append(f"missing {rel}")
            continue
        data = path.read_bytes()
        if rel.endswith((".py", ".yaml")):
            data = data.replace(b"\r\n", b"\n")
        if hashlib.sha256(data).hexdigest() != expected:
            mismatches.append(f"hash {rel}")
    if mismatches:
        raise AssertionError(f"official source differs from pinned commit: {mismatches}")
    return {
        "commit": pin["repository_commit"],
        "pin_file_sha256_lf": hashlib.sha256(Path(pin_path).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        "pinned_files_verified": len(pin["files"]),
    }


def preflight_low(low_dir, receipt_path, expected_count, smoke_one):
    """Fail closed unless low_dir holds exactly the receipt images with matching SHA256.

    Geometry is read from each low image header with PIL, the decoder the
    official code uses (main.py:284); nothing is assumed about the dataset.
    """
    import PIL.Image  # noqa: PLC0415
    receipt = json.loads(Path(receipt_path).read_bytes())
    files = receipt["files"]
    names = [item["name"] for item in files]
    if expected_count is not None:
        assert len(files) == expected_count, f"receipt has {len(files)} files, declared {expected_count}"
    assert files, "empty receipt"
    assert len(set(names)) == len(names), "duplicate receipt names"
    assert len({Path(name).stem for name in names}) == len(names), "duplicate stems"
    present = sorted(entry.name for entry in Path(low_dir).iterdir())
    assert present == sorted(names), "low directory must contain exactly the receipt images"
    geometry = {}
    for item in files:
        path = Path(low_dir) / item["name"]
        assert sha256_file(path) == item["sha256"], item["name"]
        with PIL.Image.open(path) as image:
            width, height = image.size
            orientation = image.getexif().get(274)
        # A rotation tag makes "native geometry" decoder-dependent (PIL ignores it, cv2 applies it).
        assert orientation in (None, 1), f"{item['name']}: EXIF orientation {orientation}; research-lead decision"
        geometry[item["name"]] = {"hw": [height, width], "exif_orientation": orientation}
    return files, (files[:1] if smoke_one else files), geometry


def hash_sd_snapshot(snapshot):
    snapshot = Path(snapshot)
    index = json.loads((snapshot / "model_index.json").read_bytes())
    components = sorted(key for key in index if not key.startswith("_"))
    hashed = {"model_index.json": sha256_file(snapshot / "model_index.json")}
    for component in components:
        folder = snapshot / component
        if not folder.is_dir():
            continue
        for path in sorted(p for p in folder.rglob("*") if p.is_file()):
            hashed[path.relative_to(snapshot).as_posix()] = sha256_file(path.resolve())
    unhashed = sorted(
        f"{p.name}:{p.stat().st_size}" for p in snapshot.iterdir() if p.is_file() and p.name != "model_index.json"
    )
    revision = snapshot.name if snapshot.parent.name == "snapshots" else None
    repo_dir = snapshot.parent.parent.name if revision else None
    return {
        "path": str(snapshot),
        "revision": revision,
        "hub_repo_dir": repo_dir,
        "components": components,
        "file_sha256": hashed,
        "top_level_files_not_loaded": unhashed,
    }


def vae_key_report(vae, checkpoint_path, dtype):
    """Independent recomputation of the official strict=False load (main.py:182-211)."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    raw = checkpoint["state_dict"]
    cleaned = {k.replace("my_vae.", "", 1): v for k, v in raw.items() if k.startswith("my_vae.")}
    model_state = vae.state_dict()
    missing = sorted(set(model_state) - set(cleaned))
    unexpected = sorted(set(cleaned) - set(model_state))
    mismatched = []
    for key in sorted(set(model_state) & set(cleaned)):
        loaded = model_state[key].detach().cpu()
        source = cleaned[key].to(loaded.dtype) if loaded.is_floating_point() else cleaned[key]
        if not torch.equal(loaded, source):
            mismatched.append(key)
    report = {
        "checkpoint_top_level_keys": sorted(str(k) for k in checkpoint.keys()),
        "state_dict_key_count": len(raw),
        "my_vae_key_count": len(cleaned),
        "model_key_count": len(model_state),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "loaded_value_mismatches": mismatched,
        "model_dtype": str(dtype),
    }
    del checkpoint, raw, cleaned
    return report


class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append({"level": record.levelname, "message": record.getMessage()[:4000]})


def environment_record():
    record = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "numpy": np.__version__,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "env": {k: os.environ.get(k) for k in (
            "CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF", "HF_HOME", "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE", "DIFFUSERS_OFFLINE", "XFORMERS_DISABLED")},
    }
    for name in ("diffusers", "transformers", "xformers", "accelerate", "PIL", "cv2", "einops", "huggingface_hub"):
        try:
            record[name] = __import__(name).__version__
        except Exception as exc:  # noqa: BLE001 - recorded, not fatal here
            record[name] = f"unavailable: {exc}"
    if torch.cuda.is_available():
        record["gpu"] = torch.cuda.get_device_name(0)
        record["gpu_total_bytes"] = torch.cuda.get_device_properties(0).total_memory
    return record


def save_gz(tensor, path):
    with gzip.open(path, "wb", compresslevel=1) as stream:
        torch.save(tensor, stream)


def to_tensor(hwc_rgb):
    tensor = torch.from_numpy(np.ascontiguousarray(hwc_rgb.transpose(2, 0, 1))).unsqueeze(0).contiguous()
    assert tensor.dtype == torch.float32 and torch.isfinite(tensor).all()
    return tensor


def map_back(native_rgb, low_hw, cv2):
    """Official geometry map (main.py:531-532: cv2.resize, default INTER_LINEAR, to
    the original PIL size), applied to the saturated float instead of uint8."""
    height, width = low_hw
    wide = np.clip(native_rgb.astype(np.float64), 0.0, 1.0)
    return cv2.resize(wide, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--vae-checkpoint", type=Path, required=True,
                        help="author-released QuadPrior main-epoch=00-step=7000.ckpt")
    parser.add_argument("--sd-snapshot", type=Path, required=True,
                        help="local diffusers SD1.5 snapshot directory (.../snapshots/<revision>)")
    parser.add_argument("--low-dir", type=Path, required=True)
    parser.add_argument("--low-receipt", type=Path, required=True,
                        help="JSON {files: [{name, sha256}, ...]} in canonical order")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-count", type=int,
                        help="declared cohort size; required for a promotable run")
    parser.add_argument("--smoke-one", action="store_true",
                        help="execution-only, non-promotable: first receipt image only")
    parser.add_argument("--expected-vae-sha256")
    parser.add_argument("--expected-sd-revision")
    parser.add_argument("--synthetic", action="store_true", help="synthetic probe inputs; never promotable")
    return parser


def run(args, official=None, require_cuda=True):
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    promotable = not args.smoke_one and not args.synthetic
    if promotable:
        assert args.expected_vae_sha256 and args.expected_sd_revision and args.expected_count, (
            "a promotable run requires the published --expected-vae-sha256, --expected-sd-revision "
            "and --expected-count")
    if require_cuda:
        assert torch.cuda.is_available(), "CUDA required"

    # ---------------- preflight (fail closed) ----------------
    source_record = verify_source_pin(args.source_root) if official is None else {"stub": True}
    all_files, selected, geometry = preflight_low(
        args.low_dir, args.low_receipt, args.expected_count, args.smoke_one)
    vae_sha = sha256_file(args.vae_checkpoint)
    if args.expected_vae_sha256:
        assert vae_sha == args.expected_vae_sha256, f"VAE checkpoint sha256 {vae_sha}"
    snapshot = hash_sd_snapshot(args.sd_snapshot)
    if args.expected_sd_revision:
        assert snapshot["revision"] == args.expected_sd_revision, f"SD revision {snapshot['revision']}"
    assert not args.out.exists(), "output directory must not exist"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    need = 2**30 + sum(BYTES_PER_PIXEL_BUDGET * geometry[i["name"]]["hw"][0] * geometry[i["name"]]["hw"][1]
                       for i in selected)
    free = shutil.disk_usage(args.out.parent).free
    assert free >= need, f"insufficient disk: free {free}, need {need}"
    if require_cuda:
        free_gpu, total_gpu = torch.cuda.mem_get_info()
        assert free_gpu >= 0.9 * total_gpu, f"GPU not idle: free {free_gpu} of {total_gpu}"

    # ---------------- official objects ----------------
    if official is None:
        sys.path.insert(0, str(args.source_root.resolve()))
        import main as official  # noqa: PLC0415 - official module from the pinned clone
        assert Path(official.__file__).resolve() == (args.source_root / "main.py").resolve()
    official.DEFAULT_MODEL_ID = str(args.sd_snapshot.resolve())
    handler = ListHandler()
    official.LOGGER.addHandler(handler)
    official.LOGGER.setLevel(logging.INFO)

    config = official.InferenceConfig(
        checkpoint_path=str(args.vae_checkpoint),
        img_dir_path=str(args.low_dir),
        save_dir=str(args.out / "official_output"),
        scaled_save_dir=str(args.out / "official_scaled_not_written"),
    )
    for key, value in OFFICIAL_DEFAULTS.items():
        assert getattr(config, key) == value, f"official default drift: {key}"

    args.out.mkdir(parents=False, exist_ok=False)
    load_started = time.perf_counter()
    with warnings.catch_warnings(record=True) as construction_warnings:
        warnings.simplefilter("always")
        official.seed_everything(config.seed)
        model = official.DiffusionPriorEnhancer(config)
    model_load_seconds = time.perf_counter() - load_started
    if require_cuda:
        torch.cuda.synchronize()

    failed_xformers = [r for r in handler.records if "Failed to enable xFormers" in r["message"]]
    assert not failed_xformers, f"official xFormers path not active: {failed_xformers}"
    processors = sorted({type(p).__name__ for p in model.unet.attn_processors.values()})
    assert processors and all("XFormers" in name for name in processors), processors
    encoder_module = type(model.vae.encoder).__module__
    assert encoder_module == "finetuned_vae.models", f"unexpected VAE encoder module {encoder_module}"
    key_report = vae_key_report(model.vae, args.vae_checkpoint, model.weight_dtype)
    assert not key_report["missing_keys"], "VAE missing keys: stop for research-lead review"
    assert not key_report["loaded_value_mismatches"], "loaded VAE values differ from checkpoint"
    state_before = {
        "unet": state_sha256(model.unet),
        "text_encoder": state_sha256(model.text_encoder),
        "custom_vae": state_sha256(model.vae),
    }
    scheduler_config = {k: (v if isinstance(v, (int, float, str, bool, type(None), list)) else str(v))
                        for k, v in dict(model.scheduler.config).items()}

    # ---------------- observers and input guard ----------------
    captures = []
    original_decode = model.pipeline.vae.decode
    import einops  # noqa: PLC0415 - the same function the official code uses

    def observed_decode(*call_args, **call_kwargs):
        decoded = original_decode(*call_args, **call_kwargs)
        prequant = einops.rearrange(decoded, "b c h w -> b h w c") * 127.5 + 127.5
        captures.append({"v": prequant.detach().cpu().numpy(), "dtype": str(decoded.dtype)})
        return decoded

    model.pipeline.vae.decode = observed_decode

    import cv2  # noqa: PLC0415
    import PIL.Image  # noqa: PLC0415
    allowed = {(args.low_dir / item["name"]).resolve() for item in selected}
    opened = []
    original_open = PIL.Image.open
    original_imread = cv2.imread

    def guarded_open(fp, *call_args, **call_kwargs):
        if not isinstance(fp, (str, os.PathLike)):
            raise PermissionError("only planned low image paths may be decoded")
        resolved = Path(fp).resolve()
        if resolved not in allowed:
            raise PermissionError(f"decode outside planned low images: {fp}")
        opened.append(resolved.name)
        return original_open(fp, *call_args, **call_kwargs)

    def denied_imread(*call_args, **call_kwargs):
        raise PermissionError("cv2.imread is not on the official MR. Illuminate inference path")

    PIL.Image.open = guarded_open
    cv2.imread = denied_imread

    # ---------------- per-image loop (official enumeration and order) ----------------
    official_paths = official.find_image_paths(config.img_dir_path)
    assert [p.name for p in official_paths] == [item["name"] for item in all_files], (
        "official enumeration differs from the receipt order")
    rows = []
    rng_unchanged_all = True
    first_image_warnings = []
    try:
        for index, (item, path) in enumerate(zip(selected, official_paths)):
            assert path.name == item["name"]
            low_sha = sha256_file(path)
            assert low_sha == item["sha256"], f"{path.name}: low sha256 drift"
            height, width = geometry[path.name]["hw"]
            native_hw = [-(-height // 8) * 8, -(-width // 8) * 8]
            captures.clear()
            opened.clear()
            rng_before = rng_digest()
            if require_cuda:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with warnings.catch_warnings(record=True) as image_warnings:
                warnings.simplefilter("always")
                output = model.process(path)
            if require_cuda:
                torch.cuda.synchronize()
            process_seconds = time.perf_counter() - started
            peak_allocated = torch.cuda.max_memory_allocated() if require_cuda else None
            peak_reserved = torch.cuda.max_memory_reserved() if require_cuda else None
            print(f"PEAK {path.name} allocated={peak_allocated} reserved={peak_reserved} "
                  f"process={process_seconds:.3f}s", flush=True)
            official_file = Path(official.save_output_image(config.save_dir, path, output))
            whole_seconds = time.perf_counter() - started
            rng_after = rng_digest()
            if index == 0:
                first_image_warnings = sorted({str(w.message)[:500] for w in image_warnings})

            assert opened == [path.name], f"{path.name}: decodes {opened}"
            assert len(captures) == 1, f"{path.name}: {len(captures)} decode calls"
            assert tuple(model.original_image_size) == (width, height), model.original_image_size
            resized = bool(model.should_resize_output_back)
            assert resized == (native_hw != [height, width]), "official /8 resize flag inconsistent"
            batch = captures[0]["v"]
            assert batch.shape == (1, native_hw[0], native_hw[1], 3), batch.shape
            assert np.isfinite(batch).all(), f"{path.name}: non-finite official float"
            # Recompute the official uint8 exactly as main.py:336-338 and :531-532 do.
            official_native_bgr = np.clip(batch, 0, 255).astype(np.uint8)[..., ::-1][0]
            expected_output = (cv2.resize(official_native_bgr, (width, height)) if resized
                               else official_native_bgr)
            assert isinstance(output, np.ndarray) and output.dtype == np.uint8
            assert output.shape == (height, width, 3), output.shape
            assert np.array_equal(expected_output, output), "captured float is not the official quantized image"

            prequant = batch[0]
            native_rgb = prequant.astype(np.float32) / np.float32(255.0)
            out_dir = args.out / Path(path.name).stem
            out_dir.mkdir()
            if resized:
                metric_tensor = to_tensor(map_back(native_rgb, (height, width), cv2))
                native_tensor = to_tensor(native_rgb)
                native_file = out_dir / "native_output.pt.gz"
                save_gz(native_tensor, native_file)
                native_record = {
                    "native_shape": list(native_tensor.shape),
                    "native_tensor_sha256": tensor_sha256(native_tensor),
                    "native_file_sha256": sha256_file(native_file),
                    "map_back": "cv2.resize(clip(float64(native),0,1), (W,H), INTER_LINEAR)",
                }
                del native_tensor
            else:
                metric_tensor = to_tensor(native_rgb)
                native_record = {"native_shape": [1, 3, height, width], "map_back": None}
            assert tuple(metric_tensor.shape) == (1, 3, height, width)
            output_file = out_dir / "output.pt.gz"
            save_gz(metric_tensor, output_file)
            png_file = out_dir / "official_uint8.png"
            assert cv2.imwrite(str(png_file), np.ascontiguousarray(official_native_bgr)), "PNG write failed"
            wide = prequant.astype(np.float64)
            row = {
                "low_name": path.name,
                "low_sha256": low_sha,
                "low_hw": [height, width],
                "low_exif_orientation": geometry[path.name]["exif_orientation"],
                "shape": [1, 3, height, width],
                "dtype": "torch.float32",
                "channel_order": "RGB",
                "value_definition": ("float32(v)/255, v = rearrange(decoded)*127.5+127.5 before the "
                                     "official clip/uint8; unclipped unless map_back is set"),
                "output_tensor_sha256": tensor_sha256(metric_tensor),
                "output_file_sha256": sha256_file(output_file),
                "official_resize_back": resized,
                **native_record,
                "prequant_dtype": captures[0]["dtype"],
                "prequant_min": float(wide.min()),
                "prequant_max": float(wide.max()),
                "prequant_fraction_below_0": float((wide < 0).mean()),
                "prequant_fraction_above_255": float((wide > 255).mean()),
                "official_native_uint8_bgr_sha256": hashlib.sha256(
                    np.ascontiguousarray(official_native_bgr).tobytes()).hexdigest(),
                "official_uint8_match": True,
                "official_uint8_png": f"{Path(path.name).stem}/official_uint8.png",
                "official_uint8_png_sha256": sha256_file(png_file),
                "official_output_file": f"official_output/{official_file.name}",
                "official_output_file_sha256": sha256_file(official_file),
                "global_rng_unchanged": rng_before == rng_after,
                "process_seconds": process_seconds,
                "whole_run_seconds": whole_seconds,
                "peak_gpu_memory_allocated_bytes": peak_allocated,
                "peak_gpu_memory_bytes": peak_reserved,
            }
            rng_unchanged_all = rng_unchanged_all and row["global_rng_unchanged"]
            (out_dir / "decision.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
            rows.append(row)
            del metric_tensor, native_rgb, wide, prequant, batch, output
            captures.clear()
            print(f"{index + 1}/{len(selected)} {path.name} {whole_seconds:.3f}s", flush=True)
    finally:
        PIL.Image.open = original_open
        cv2.imread = original_imread
        model.pipeline.vae.decode = original_decode

    assert len(rows) == len(selected), "row count mismatch"
    written = sorted(p.name for p in (args.out / "official_output").iterdir())
    assert written == sorted(item["name"] for item in selected), "official output set mismatch"
    state_after = {
        "unet": state_sha256(model.unet),
        "text_encoder": state_sha256(model.text_encoder),
        "custom_vae": state_sha256(model.vae),
    }
    unchanged = state_after == state_before
    assert unchanged, "model state changed during inference"
    manifest = {
        "method": "MR-Illuminate-official-native" + ("" if promotable else "-NONFINAL"),
        "promotable": promotable,
        "smoke_one": args.smoke_one,
        "synthetic": args.synthetic,
        "declared_count": args.expected_count,
        "count": len(rows),
        "started_utc": started_utc,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runner_sha256_lf": self_sha256_lf(),
        "source": source_record,
        "low_dir": str(args.low_dir),
        "low_receipt_sha256": sha256_file(args.low_receipt),
        "vae_checkpoint": {"path": str(args.vae_checkpoint), "sha256": vae_sha,
                           "bytes": args.vae_checkpoint.stat().st_size},
        "checkpoint_sha256": vae_sha,
        "sd_snapshot": snapshot,
        "vae_key_report": key_report,
        "official_config": {k: getattr(config, k) for k in (
            "seed", "target_avg", "n_timesteps", "guidance_scale", "prompt", "negative_prompt",
            "enable_xformers", "device")},
        "weight_dtype": str(model.weight_dtype),
        "scheduler_config": scheduler_config,
        "adain_generator_seed": 145,
        "attention_processors": processors,
        "model_load_seconds": model_load_seconds,
        "official_log_records": handler.records,
        "construction_warnings": sorted({str(w.message)[:500] for w in construction_warnings}),
        "first_image_warnings": first_image_warnings,
        "environment": environment_record(),
        "model_state_sha256": state_before,
        "model_state_unchanged": unchanged,
        "global_rng_unchanged_all_images": rng_unchanged_all,
        "execution_only_differences": [
            "no per-image exception swallowing (official main.py:614)",
            "brightness-scaled input baseline not written (main.py:607-613)",
            "DEFAULT_MODEL_ID -> pinned local SD1.5 snapshot, hub offline",
            "observer on custom VAE decode; PIL decode guard; cv2.imread denied",
        ],
        "reference_reads": 0,
        "metrics": 0,
        "rows": rows,
    }
    manifest_path = args.out / "output_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"OUTPUT_MANIFEST_SHA256 {sha256_file(manifest_path)}", flush=True)
    return manifest


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
