# T073-E QuadPrior low-only execution gate

2026-09-25, written before any model execution on any dataset. Status: **runner, verifier and CPU tests ready; blocked on the three official weights and the `quadprior` env** (torch 1.12.1+cu113 is present; pip installs are still running). Target model runs 0, reference reads 0, metrics 0. Scope change (T074-A): the runner is dataset-generic and must not be run on UHD-LL target images.

## What is official (unmodified, pinned)

- Source: daooshee/QuadPrior `cbdf02f2`. `quadprior_source_pin.json` pins 55 `.py`/`.yaml` files plus `empty_embedding.pkl` (`322aef8a…`). The staged host copy verified 56/56.
- **`test.py` itself is executed** via `runpy` as `__main__`. `process` cannot be imported, because everything sits under `__main__`. Weight loading follows the file:
  - `control_sd15_ini` loaded with strict=True;
  - `add_new_layers`;
  - `COCO-final` loaded into `control_model` with strict=True;
  - `change_first_stage` (bypass VAE, strict=True);
  - `.cuda().to(float16)`.

  Per image:
  - `cv2.imread` (BGR);
  - `resize_image(HWC3, 512)`: short side 512, both sides rounded to multiples of 64, INTER_AREA when downscaling (annotator/util.py:28-37);
  - `seed_everything(0)`;
  - DPM-Solver++ with order 3, 10 steps, scale 9 (a no-op), eta 0;
  - `decode_new_first_stage`.

  Only `--checkpoint`, `--same_folder` and `--input_folder` are passed. The runner asserts `use_float16=True` and `save_memory=False`.

## What is execution-only

1. **Task-owned cwd (`--workdir`).** It holds symlinks for the cwd-relative paths the official code opens: `models/cldm_v15.yaml`, `models/control_sd15_ini.ckpt`, `checkpoints/main-epoch=00-step=7000.ckpt`, and `empty_embedding.pkl` (cldm.py:326). It also holds an `input/` folder of symlinks to exactly the planned receipt images. The official clone keeps its zero-byte placeholders.
2. **Hooks.** All are removed in `finally`.
   - `cv2.imread` admits only the planned files, re-hashes each one, and starts the timer.
   - `LatentDiffusion.decode_new_first_stage` returns the result unchanged and records `rearrange(x)*127.5+127.5` (test.py:116).
   - `cv2.imwrite` writes only to the official output folder, then asserts that `clip(v).astype(uint8)` equals the saved array bitwise and writes the float artifacts.
   - `DPMSolverSampler.__init__` hashes the fully loaded fp16 state; the hash is compared after the run.
   - `PIL.Image.open` is denied.
3. **No silent skip.** Output stems must be unique, the output directory must be new, every planned image must pass through imread→decode→imwrite exactly once, and the official folder must hold exactly N PNGs.

## Output format and geometry (decisions 1-2)

- `native_output.pt.gz`: float32 `[1,3,h,w]` RGB, equal to `float32(v[..., ::-1])/255`. It is the official float before clip/uint8, at the official geometry (for example 512×896 for 16:9). The flip is needed because the official array is BGR: control is fed as BGR and the result is saved with cv2.
- `output.pt.gz` (metric artifact): float32 `[1,3,H,W]` RGB, equal to `cv2.resize(clip(float64(native),0,1), (W,H), INTER_LINEAR)`. H and W are this low image's own cv2-decoded geometry. This is the only official map to evaluation geometry (bilinear `cv2.resize` to the target size, paired-metrics.py:72-74), applied in the official order (saturate, then resize) without the 8-bit quantization. It is applied before freezing, so no reference size is needed.
- Provenance: the official PNG in `official_output/`.

## Verifier (`verify_quadprior_full.py`)

It checks:

- receipt order and hashes; manifest/decision equality;
- the three weight pins; the official args;
- that the processing order is a permutation of the receipt;
- that the output geometry equals the low image's cv2 geometry, and that the native geometry equals the official formula;
- tensor and file hashes; finiteness;
- the native float against the official PNG truncation bin, with slack 1e-3;
- the map-back recomputation, to within 1e-6;
- `reference_reads`/`metrics` equal to 0 and an unchanged state;
- optionally, the first-row hash against the smoke output.

## Tests (CPU, host base env)

`test_quadprior_runner_and_verifier.py`: 13/13 pass. A fake source tree mirrors the hooked module paths, and a fake `test.py` copies the official resize/quantize/save code. It runs on 20×36 (native 512×896) and 24×24 (native 512×512) images.

- Runner tests cover: the happy path plus the verifier, including the RGB order check and native unclipped vs output saturated; unplanned read denied; tampered uint8; stray low file; an existing output directory; a promotable run without pins; smoke-one.
- Verifier mutations cover: tensor hash; map-back; bin violation; low mutation; weight pin; missing official PNG.
- Both runners also compile under the target Python versions: 3.8 for quadprior, 3.9 for mri.

## Pending research-lead decisions

1. Keep the official 512-short-side inference plus bilinear map-back to the low geometry, as implemented. The alternative is an unofficial native-resolution run. Disclose the aspect change from rounding to multiples of 64: about 1.6% at 16:9.
2. The same cross-row metric-time clip rule as in T073-D. This row's metric artifact is already in [0,1].
3. cv2 applies EXIF orientation and PIL does not, so the runner stops on any orientation tag outside {None, 1}.

## Stop conditions

Any assertion (pins, receipt, EXIF, geometry, official-uint8 identity, non-finite value, a skip or stray file, state drift, `use_float16`), an OOM, or a nonzero exit. No automatic change of precision, resolution, seed or steps. Note that `--use_float16` cannot be disabled from the official CLI anyway (test.py:25).

## Execution order once unblocked

1. `probe_quadprior_synthetic.py` with the real weights: peak VRAM (README: about 13-14 GB), repeat determinism, prequant dtype.
2. Publish the three weight SHA256s.
3. `--smoke-one` on the chosen dataset, once authorized.
4. The full run with `--expected-count` and all three `--expected-*-sha256`.
5. `verify_quadprior_full.py --require-promotable --expected-output-manifest-sha256 <logged>`.
