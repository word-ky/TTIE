"""Dataset-generic low-only official QuadPrior runner (T073-E).

Executes the unmodified official `test.py` (commit cbdf02f2) with `runpy` as
`__main__`, so model construction, weight loading (all strict), fp16 cast,
`resize_image(HWC3(img), 512)`, per-image `seed_everything(0)`, DPM-Solver++
(order 3, 10 steps, scale 9, eta 0) and `decode_new_first_stage` are the
official code text (test.py:37-120). Official defaults are not overridden;
only `--checkpoint`, `--same_folder` and `--input_folder` are passed.

Execution-only plumbing:

* a task-owned working directory supplies the cwd-relative files test.py and
  cldm.py:326 open (`models/cldm_v15.yaml`, `models/control_sd15_ini.ckpt`,
  `checkpoints/main-epoch=00-step=7000.ckpt`, `empty_embedding.pkl`) as
  symlinks to the pinned source and the hashed weights, plus an `input/`
  folder holding symlinks to exactly the planned receipt low images;
* observers: `cv2.imread` admits only those planned files (anything else is
  denied) and starts the per-image timer; `LatentDiffusion.decode_new_first_stage`
  returns the official result unchanged and records the same pre-quantization
  float test.py:116 computes; `cv2.imwrite` writes only into the official
  output folder and then finalizes the record; `DPMSolverSampler.__init__`
  hashes the fully loaded model state; `PIL.Image.open` is denied.

Frozen artifacts per image (float32, RGB, `[1, 3, h, w]`):
* `native_output.pt.gz`: `float32(v[..., ::-1]) / 255` at the official
  512-short-side geometry, `v = rearrange(x_samples) * 127.5 + 127.5` before
  the official clip/uint8 (unclipped; v is BGR because test.py saves with cv2);
* `output.pt.gz` (metric artifact): the only official map to evaluation
  geometry (paired-metrics.py:72-74, bilinear `cv2.resize` to the target size)
  applied to `clip(float64(native), 0, 1)` with dsize = this low image's own
  decoded geometry. The official order is saturate, then resize; only the
  8-bit quantization is omitted.
The official PNG written by test.py is kept for provenance. The runner asserts
bitwise that `clip(v, 0, 255).astype(uint8)` equals the array test.py saved.

No reference/GT path exists in this program; it computes no metric.
"""

import argparse
import gzip
import hashlib
import json
import os
import platform
import runpy
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PIN_FILE = HERE / "quadprior_source_pin.json"
IMAGE_RESOLUTION = 512  # official process() default, test.py:68
BYTES_PER_PIXEL_BUDGET = 16
HOOKS = {
    "decode_owner": ("ldm.models.diffusion.ddpm", "LatentDiffusion"),
    "sampler": ("ldm.models.diffusion.dpm_solver", "DPMSolverSampler"),
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


def capture_state_refs(module):
    """Name -> tensor references taken once, before any image, while the graph is acyclic.
    The post-run check re-hashes these references rather than calling state_dict() again,
    which would recurse forever if anything registered a parent module as a child."""
    return dict(module.state_dict(keep_vars=True))


def tensor_identity(module):
    """Identity sets via parameters()/buffers() (memoised traversal, cycle-safe)."""
    return {"parameters": frozenset(id(t) for t in module.parameters()),
            "buffers": frozenset(id(t) for t in module.buffers())}


def state_hashes(refs):
    """One pass over captured references: total digest plus one digest per top-level child."""
    total = hashlib.sha256()
    parts = {}
    for key, value in sorted(refs.items()):
        array = value.detach().cpu().contiguous()
        blob = key.encode() + str(array.dtype).encode() + str(tuple(array.shape)).encode()
        data = array.numpy().tobytes()
        for digest in (total, parts.setdefault(key.split(".")[0], hashlib.sha256())):
            digest.update(blob)
            digest.update(data)
    return {"total": total.hexdigest(), **{k: v.hexdigest() for k, v in sorted(parts.items())}}


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


def official_native_hw(height, width, resolution=IMAGE_RESOLUTION):
    """Arithmetic of annotator/util.py:28-37 (resize_image), used as a cross-check."""
    h, w = float(height), float(width)
    k = float(resolution) / min(h, w)
    h *= k
    w *= k
    return [int(np.round(h / 64.0)) * 64, int(np.round(w / 64.0)) * 64]


def map_back(native_rgb, low_hw, cv2):
    """paired-metrics.py:74 (cv2.resize, default INTER_LINEAR, float64 input) on the
    saturated official float instead of the 8-bit file."""
    height, width = low_hw
    wide = np.clip(native_rgb.astype(np.float64), 0.0, 1.0)
    return cv2.resize(wide, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def to_tensor(hwc_rgb):
    tensor = torch.from_numpy(np.ascontiguousarray(hwc_rgb.transpose(2, 0, 1))).unsqueeze(0).contiguous()
    assert tensor.dtype == torch.float32 and torch.isfinite(tensor).all()
    return tensor


def save_gz(tensor, path):
    with gzip.open(path, "wb", compresslevel=1) as stream:
        torch.save(tensor, stream)


def preflight_low(low_dir, receipt_path, expected_count, smoke_one):
    """Exact receipt set in low_dir, SHA256 match, geometry from the cv2 header decode."""
    import cv2  # noqa: PLC0415
    receipt = json.loads(Path(receipt_path).read_bytes())
    files = receipt["files"]
    names = [item["name"] for item in files]
    if expected_count is not None:
        assert len(files) == expected_count, f"receipt has {len(files)} files, declared {expected_count}"
    assert files, "empty receipt"
    assert len(set(names)) == len(names), "duplicate receipt names"
    # test.py:124-126 names outputs <stem>.png and skips existing ones: stems must be unique.
    assert len({Path(name).stem for name in names}) == len(names), "duplicate stems"
    assert all("." in name and not name.startswith(".") for name in names), "glob('*.*') would miss a name"
    present = sorted(entry.name for entry in Path(low_dir).iterdir())
    assert present == sorted(names), "low directory must contain exactly the receipt images"
    for item in files:
        assert sha256_file(Path(low_dir) / item["name"]) == item["sha256"], item["name"]
    selected = files[:1] if smoke_one else files
    geometry = {}
    import PIL.Image  # noqa: PLC0415 - header only, before any hook is installed
    for item in selected:
        with PIL.Image.open(Path(low_dir) / item["name"]) as header:
            orientation = header.getexif().get(274)
        # A rotation tag makes "native geometry" decoder-dependent (cv2 applies it, PIL does not).
        assert orientation in (None, 1), f"{item['name']}: EXIF orientation {orientation}; research-lead decision"
        image = cv2.imread(str(Path(low_dir) / item["name"]))
        assert image is not None and image.ndim == 3 and image.shape[2] == 3, item["name"]
        geometry[item["name"]] = list(image.shape[:2])
    return files, selected, geometry


def build_workdir(workdir, source_root, control_ckpt, vae_ckpt, low_dir, selected):
    workdir.mkdir(parents=True, exist_ok=False)
    for sub in ("models", "checkpoints", "input"):
        (workdir / sub).mkdir()
    links = {
        "models/cldm_v15.yaml": source_root / "models" / "cldm_v15.yaml",
        "models/control_sd15_ini.ckpt": control_ckpt,
        "checkpoints/main-epoch=00-step=7000.ckpt": vae_ckpt,
        "empty_embedding.pkl": source_root / "empty_embedding.pkl",
    }
    for item in selected:
        links[f"input/{item['name']}"] = low_dir / item["name"]
    for rel, target in links.items():
        os.symlink(Path(target).resolve(), workdir / rel)
    return {rel: str(Path(target).resolve()) for rel, target in links.items() if not rel.startswith("input/")}


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
        "env": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF")},
    }
    for name in ("pytorch_lightning", "transformers", "deepspeed", "cv2", "einops", "omegaconf", "xformers"):
        try:
            record[name] = __import__(name).__version__
        except Exception as exc:  # noqa: BLE001 - recorded, not fatal here
            record[name] = f"unavailable: {exc}"
    if torch.cuda.is_available():
        record["gpu"] = torch.cuda.get_device_name(0)
        record["gpu_total_bytes"] = torch.cuda.get_device_properties(0).total_memory
    return record


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--coco-checkpoint", type=Path, required=True, help="official COCO-final.ckpt")
    parser.add_argument("--vae-checkpoint", type=Path, required=True,
                        help="official main-epoch=00-step=7000.ckpt")
    parser.add_argument("--control-checkpoint", type=Path, required=True,
                        help="official control_sd15_ini.ckpt")
    parser.add_argument("--low-dir", type=Path, required=True)
    parser.add_argument("--low-receipt", type=Path, required=True,
                        help="JSON {files: [{name, sha256}, ...]} in canonical order")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True, help="task-owned cwd; must not exist")
    parser.add_argument("--expected-count", type=int, help="declared cohort size; required for a promotable run")
    parser.add_argument("--smoke-one", action="store_true",
                        help="execution-only, non-promotable: first receipt image only")
    parser.add_argument("--expected-coco-sha256")
    parser.add_argument("--expected-vae-sha256")
    parser.add_argument("--expected-control-sha256")
    parser.add_argument("--synthetic", action="store_true", help="synthetic probe inputs; never promotable")
    return parser


def run(args, require_cuda=True, pin_path=PIN_FILE):
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    promotable = not args.smoke_one and not args.synthetic
    if promotable:
        assert (args.expected_coco_sha256 and args.expected_vae_sha256 and args.expected_control_sha256
                and args.expected_count), "a promotable run requires all three weight pins and --expected-count"
    if require_cuda:
        assert torch.cuda.is_available(), "CUDA required"
    source_root = args.source_root.resolve()
    # test.py runs with the task-owned cwd, so every path used in hooks must be absolute.
    for key in ("low_dir", "low_receipt", "out", "workdir", "coco_checkpoint", "vae_checkpoint",
                "control_checkpoint"):
        setattr(args, key, Path(os.path.abspath(getattr(args, key))))

    # ---------------- preflight (fail closed) ----------------
    source_record = verify_source_pin(source_root, pin_path) if pin_path else {"pin": None}
    all_files, selected, geometry = preflight_low(args.low_dir, args.low_receipt, args.expected_count,
                                                  args.smoke_one)
    weights = {}
    for label, path, expected in (
            ("coco_final", args.coco_checkpoint, args.expected_coco_sha256),
            ("vae_main_epoch00_step7000", args.vae_checkpoint, args.expected_vae_sha256),
            ("control_sd15_ini", args.control_checkpoint, args.expected_control_sha256)):
        digest = sha256_file(path)
        if expected:
            assert digest == expected, f"{label} sha256 {digest}"
        weights[label] = {"path": str(path), "sha256": digest, "bytes": Path(path).stat().st_size}
    assert not args.out.exists(), "output directory must not exist"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    need = 2**30 + sum(BYTES_PER_PIXEL_BUDGET * h * w for h, w in geometry.values())
    free = shutil.disk_usage(args.out.parent).free
    assert free >= need, f"insufficient disk: free {free}, need {need}"
    if require_cuda:
        free_gpu, total_gpu = torch.cuda.mem_get_info()
        assert free_gpu >= 0.9 * total_gpu, f"GPU not idle: free {free_gpu} of {total_gpu}"
    args.out.mkdir()
    official_dir = args.out / "official_output"
    links = build_workdir(args.workdir, source_root, args.control_checkpoint, args.vae_checkpoint,
                          args.low_dir, selected)

    # ---------------- hooks ----------------
    sys.path.insert(0, str(source_root))
    import importlib  # noqa: PLC0415
    import cv2  # noqa: PLC0415
    import einops  # noqa: PLC0415 - the same function test.py uses
    import PIL.Image  # noqa: PLC0415
    decode_owner = getattr(importlib.import_module(HOOKS["decode_owner"][0]), HOOKS["decode_owner"][1])
    sampler_cls = getattr(importlib.import_module(HOOKS["sampler"][0]), HOOKS["sampler"][1])
    assert str(Path(sys.modules[HOOKS["decode_owner"][0]].__file__).resolve()).startswith(str(source_root) + os.sep)

    # Official non-xformers attention path: each of these modules defines the same
    # try/except XFORMERS_IS_AVAILBLE flag (official spelling) at import time
    # (ldm/modules/attention.py:13-17, ldm/modules/diffusionmodules/model.py:12-14,
    # my_vae/models.py:12-16). ATTN_PRECISION governs attention.py's cross-attn softmax.
    attention_modules = {
        "ldm.modules.attention": importlib.import_module("ldm.modules.attention"),
        "ldm.modules.diffusionmodules.model": importlib.import_module("ldm.modules.diffusionmodules.model"),
        "my_vae.models": importlib.import_module("my_vae.models"),
    }
    attention_backend = {f"{name}.XFORMERS_IS_AVAILBLE": mod.XFORMERS_IS_AVAILBLE
                         for name, mod in attention_modules.items()}
    for key, value in attention_backend.items():
        assert value is False, f"{key} is True: official non-xformers attention path not active"
    attention_backend["ATTN_PRECISION"] = os.environ.get("ATTN_PRECISION", "fp32")
    assert attention_backend["ATTN_PRECISION"] == "fp32", (
        f"ATTN_PRECISION={attention_backend['ATTN_PRECISION']!r}, official default is fp32")

    input_dir = (args.workdir / "input").resolve()
    planned = {item["name"]: item for item in selected}
    state = {"current": None, "model": None, "state_before": None, "load_seconds": None,
             "weights_peak": None}
    captures, rows_by_name, order = [], {}, []
    original = {
        "imread": cv2.imread, "imwrite": cv2.imwrite, "open": PIL.Image.open,
        "decode": decode_owner.decode_new_first_stage, "sampler_init": sampler_cls.__init__,
    }
    run_started = time.perf_counter()

    def sync():
        if require_cuda:
            torch.cuda.synchronize()

    def guarded_imread(path, *call_args, **call_kwargs):
        candidate = Path(path)
        if (candidate.parent.resolve() != input_dir or candidate.name not in planned
                or candidate.resolve() != (args.low_dir / candidate.name).resolve()):
            raise PermissionError(f"decode outside planned low images: {path}")
        if candidate.name in rows_by_name or state["current"] is not None:
            raise RuntimeError(f"{candidate.name}: repeated or overlapping image")
        low_sha = sha256_file(candidate.resolve())
        assert low_sha == planned[candidate.name]["sha256"], f"{candidate.name}: low sha256 drift"
        sync()
        if require_cuda:
            torch.cuda.reset_peak_memory_stats()
        state["current"] = {"name": candidate.name, "low_sha256": low_sha, "t0": time.perf_counter()}
        image = original["imread"](path, *call_args, **call_kwargs)
        state["current"]["input_hw"] = list(image.shape[:2])
        order.append(candidate.name)
        return image

    def observed_decode(self, *call_args, **call_kwargs):
        decoded = original["decode"](self, *call_args, **call_kwargs)
        prequant = einops.rearrange(decoded, "b c h w -> b h w c") * 127.5 + 127.5
        captures.append({"v": prequant.cpu().numpy(), "dtype": str(decoded.dtype)})
        return decoded

    def observed_sampler_init(self, model, *call_args, **call_kwargs):
        original["sampler_init"](self, model, *call_args, **call_kwargs)
        sync()
        state["load_seconds"] = time.perf_counter() - run_started
        state["weights_peak"] = torch.cuda.max_memory_reserved() if require_cuda else None
        state["model"] = model
        state["state_refs"] = capture_state_refs(model)
        state["state_before"] = state_hashes(state["state_refs"])
        state["identity_before"] = tensor_identity(model)
        hashed_ids = {id(t) for t in state["state_refs"].values()}
        live = state["identity_before"]["parameters"] | state["identity_before"]["buffers"]
        assert hashed_ids <= live, "hashed tensors are not the model's live parameters/buffers"
        state["first_stage_encoder_module"] = type(model.first_stage_model.encoder).__module__

    def guarded_imwrite(path, image, *call_args, **call_kwargs):
        target = Path(path)
        if target.parent.resolve() != official_dir.resolve():
            raise PermissionError(f"write outside the official output folder: {path}")
        current = state["current"]
        assert current is not None, "imwrite without a current image"
        ok = original["imwrite"](path, image, *call_args, **call_kwargs)
        assert ok, f"official imwrite failed: {path}"
        sync()
        process_seconds = time.perf_counter() - current["t0"]
        peak_allocated = torch.cuda.max_memory_allocated() if require_cuda else None
        peak_reserved = torch.cuda.max_memory_reserved() if require_cuda else None
        print(f"PEAK {current['name']} allocated={peak_allocated} reserved={peak_reserved} "
              f"process={process_seconds:.3f}s", flush=True)
        name = current["name"]
        assert target.name == Path(name).stem + ".png", target.name
        assert len(captures) == 1, f"{name}: {len(captures)} decode calls"
        capture = captures.pop()
        height, width = current["input_hw"]
        assert [height, width] == geometry[name], f"{name}: decoded geometry differs from preflight"
        native_hw = official_native_hw(height, width)
        batch = capture["v"]
        assert batch.shape == (1, native_hw[0], native_hw[1], 3), batch.shape
        assert np.isfinite(batch).all(), f"{name}: non-finite official float"
        official_u8 = np.clip(batch, 0, 255).astype(np.uint8)[0]
        assert isinstance(image, np.ndarray) and image.dtype == np.uint8
        assert np.array_equal(official_u8, image), "captured float is not the official quantized image"
        prequant = batch[0]
        native_rgb = prequant[..., ::-1].astype(np.float32) / np.float32(255.0)
        native_tensor = to_tensor(native_rgb)
        metric_tensor = to_tensor(map_back(native_rgb, (height, width), cv2))
        assert tuple(metric_tensor.shape) == (1, 3, height, width)
        out_dir = args.out / Path(name).stem
        out_dir.mkdir()
        native_file = out_dir / "native_output.pt.gz"
        output_file = out_dir / "output.pt.gz"
        save_gz(native_tensor, native_file)
        save_gz(metric_tensor, output_file)
        wide = prequant.astype(np.float64)
        row = {
            "low_name": name,
            "low_sha256": current["low_sha256"],
            "low_hw": [height, width],
            "shape": [1, 3, height, width],
            "dtype": "torch.float32",
            "channel_order": "RGB",
            "output_tensor_sha256": tensor_sha256(metric_tensor),
            "output_file_sha256": sha256_file(output_file),
            "map_back": "cv2.resize(clip(float64(native),0,1), (W,H), INTER_LINEAR)",
            "native_shape": list(native_tensor.shape),
            "native_tensor_sha256": tensor_sha256(native_tensor),
            "native_file_sha256": sha256_file(native_file),
            "native_value_definition": ("float32(v[..., ::-1])/255, v = rearrange(x_samples)*127.5+127.5 "
                                        "before the official clip/uint8; unclipped"),
            "prequant_dtype": capture["dtype"],
            "prequant_min": float(wide.min()),
            "prequant_max": float(wide.max()),
            "prequant_fraction_below_0": float((wide < 0).mean()),
            "prequant_fraction_above_255": float((wide > 255).mean()),
            "official_uint8_bgr_sha256": hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest(),
            "official_uint8_match": True,
            "official_output_file": f"official_output/{target.name}",
            "official_output_file_sha256": sha256_file(target),
            "official_processing_index": len(order) - 1,
            "process_seconds": process_seconds,
            "whole_run_seconds": process_seconds,
            "peak_gpu_memory_allocated_bytes": peak_allocated,
            "peak_gpu_memory_bytes": peak_reserved,
        }
        (out_dir / "decision.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        rows_by_name[name] = row
        state["current"] = None
        print(f"{len(rows_by_name)}/{len(selected)} {name} {process_seconds:.3f}s", flush=True)
        return ok

    def denied_open(*call_args, **call_kwargs):
        raise PermissionError("PIL decode is not on the official QuadPrior inference path")

    saved_argv, saved_cwd = sys.argv, os.getcwd()
    cv2.imread, cv2.imwrite, PIL.Image.open = guarded_imread, guarded_imwrite, denied_open
    decode_owner.decode_new_first_stage = observed_decode
    sampler_cls.__init__ = observed_sampler_init
    sys.argv = [str(source_root / "test.py"), "--checkpoint", str(args.coco_checkpoint.resolve()),
                "--same_folder", str(official_dir.resolve()), "--input_folder", "input"]
    try:
        os.chdir(args.workdir)
        final_globals = runpy.run_path(str(source_root / "test.py"), run_name="__main__")
    finally:
        os.chdir(saved_cwd)
        sys.argv = saved_argv
        cv2.imread, cv2.imwrite, PIL.Image.open = original["imread"], original["imwrite"], original["open"]
        decode_owner.decode_new_first_stage = original["decode"]
        sampler_cls.__init__ = original["sampler_init"]

    # ---------------- completeness and state (fail closed) ----------------
    assert state["current"] is None and not captures, "an image started but was not saved"
    assert sorted(rows_by_name) == sorted(planned), "not every planned image was processed exactly once"
    assert sorted(p.name for p in official_dir.iterdir()) == sorted(Path(n).stem + ".png" for n in planned), (
        "official output folder differs from the plan (skip or stray file)")
    assert state.get("first_stage_encoder_module") == "my_vae.models", state.get("first_stage_encoder_module")
    official_args = vars(final_globals["args"])
    assert official_args["use_float16"] is True and official_args["save_memory"] is False, official_args
    model = final_globals["model"]
    assert model is state["model"], "sampler hook saw a different model"
    state_after = state_hashes(state["state_refs"])
    identity_unchanged = tensor_identity(model) == state["identity_before"]
    assert identity_unchanged, "parameter/buffer tensor objects were replaced during inference"
    unchanged = state_after == state["state_before"] and identity_unchanged
    assert unchanged, "model state changed during inference"
    rows = [rows_by_name[item["name"]] for item in selected]
    manifest = {
        "method": "QuadPrior-official-test-py" + ("" if promotable else "-NONFINAL"),
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
        "weights": weights,
        "workdir": str(args.workdir),
        "workdir_links": links,
        "official_args": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                          for k, v in official_args.items()},
        "official_process_defaults": {"image_resolution": IMAGE_RESOLUTION, "diffusion_steps": 10,
                                      "scale": 9.0, "seed_per_image": 0, "eta": 0.0, "strength": 1.0,
                                      "guess_mode": False, "dmp_order": 3, "num_samples": 1},
        "official_processing_order": order,
        "attention_backend": attention_backend,
        "first_stage_encoder_module": state.get("first_stage_encoder_module"),
        "model_load_seconds": state["load_seconds"],
        "weights_loaded_peak_reserved_bytes": state["weights_peak"],
        "environment": environment_record(),
        "model_state_sha256": state["state_before"],
        "model_state_unchanged": unchanged,
        "model_state_check": ("state_dict(keep_vars=True) references captured at sampler construction, "
                              "re-hashed after the run; parameters()/buffers() identity sets unchanged"),
        "execution_only_differences": [
            "task-owned cwd with symlinks for cwd-relative official paths and a planned-only input folder",
            "observers on cv2.imread/cv2.imwrite/decode_new_first_stage/DPMSolverSampler.__init__; PIL denied",
            "float artifacts and bilinear map-back written in addition to the official PNG",
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
