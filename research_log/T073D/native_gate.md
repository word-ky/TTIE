# T073-D MR. Illuminate low-only native execution gate

2026-09-25, written before any model execution on any dataset. Status: **runner, verifier and CPU tests ready; blocked on weights and the `mri` env**. Target model runs 0, reference reads 0, metrics 0. Scope change (T074-A): UHD-LL is no longer a main-table target. The runner is dataset-generic and must not be run on UHD-LL target images.

## What is official (unmodified, pinned)

- Source: joshua-cho-story/MR-Illuminate `013e013e`. `mri_source_pin.json` pins 48 `.py`/`.yaml` blobs. The runner refuses a source tree whose CRLF-normalised hashes differ. The staged host copy `/root/autodl-tmp/TTIE/T073D/source/MR-Illuminate` (CRLF) verified 48/48.
- The model path is called, not copied: `seed_everything(1)`, then `DiffusionPriorEnhancer(config)` with the `InferenceConfig` defaults (seed 1, target_avg 30, 25 DDIM steps, empty prompt, xFormers on; asserted), then `find_image_paths` (sorted), then `process`, then `save_output_image`. This is the same order as `run_inference` (main.py:588-620). Preprocessing is official: PIL RGB decode; LANCZOS up to a multiple of 8 only when needed (main.py:291-299); mean<30 brightening (main.py:304); `posterior.mean` encode; DDIM inversion plus self-attention injection; AdaIN init with generator seed 145 (main.py:456); custom-VAE decode with encoder skips.

## What is execution-only (and why it cannot change outputs)

1. **Fail-closed loop.** Per-image exceptions are not swallowed (official main.py:614 does swallow them). Exactly N rows are required, and the official output folder must hold exactly the receipt names.
2. **Scaled baseline not written.** main.py:607-613 is a brightness-scaled copy of the input, not the method output. It uses no model state and no RNG.
3. **SD1.5 location.** `main.DEFAULT_MODEL_ID` (read at call time, main.py:132,155) points to a local diffusers snapshot. The hub is forced offline (`HF_HUB_OFFLINE=1` etc. are set before any import). The runner records the revision, the SHA256 of every component file, and per-module loaded-state hashes (unet, text_encoder, custom VAE).
4. **Observer on the custom VAE's bound `decode`.** It returns the official tensor unchanged. It also computes the identical expression `rearrange(decoded)*127.5+127.5` on the same device and dtype (main.py:333-335) and copies it to the CPU.
5. **Input guard.** `PIL.Image.open` accepts only the planned receipt files. Exactly one decode per image is asserted. `cv2.imread` is denied.
6. **Independent check of the `strict=False` VAE load.** Missing, unexpected and value-mismatched keys are recomputed from the checkpoint and recorded. The runner also asserts that the VAE encoder class comes from `finetuned_vae.models`, because the `modules.models` try-import could shadow it.

## Output format (decision 1)

`<stem>/output.pt.gz`: a gzip `torch.save` of a float32 `[1,3,H,W]` RGB tensor, the same container as the frozen PromptIR/DCTTA rows. H and W are the low image's own PIL geometry.

- When both sides are multiples of 8: `float32(v)/255`, with `v` the official pre-quantization float, **unclipped**. This follows the PromptIR precedent of storing raw values and deferring any clip.
- Otherwise the official code maps its uint8 back with `cv2.resize` (main.py:531-532). The runner applies the same map, INTER_LINEAR, to `clip(float32(v)/255, 0, 1)`: the official order is saturate, then resize, with only the 8-bit quantization omitted. The native float is also kept in `native_output.pt.gz`.
- Before saving, the runner asserts bitwise that the official uint8 recomputed from `v` (including the resize-back) equals what `process` returned.
- Provenance only, never used for metrics: `official_uint8.png` (lossless, native geometry, BGR) and the official file in `official_output/`. The official file is lossy JPEG for `.jpg` inputs, because the official saver keeps the input extension.

## Verifier (`verify_mri_full.py`)

It re-hashes the low images against the receipt and checks:

- manifest/decision equality and receipt order;
- tensor and file SHA256; shape equal to the low image's PIL geometry; float32; finite;
- that every native float lies in the truncation bin of the official uint8, with slack 1e-3 in the 0-255 scale;
- the map-back recomputation, to within 1e-6;
- the pins (VAE SHA256, SD revision), an empty missing-key list, `reference_reads`/`metrics` equal to 0 and an unchanged model state;
- optionally, the first-row tensor hash against the smoke output.

## Tests (CPU, host base env torch 2.3.0)

`test_mri_runner_and_verifier.py`: 16/16 pass. A stub official module reproduces main.py's quantize and /8 resize-back code on a 8×16 image and a 10×18 image. Runner tests cover: the happy path plus the verifier; stray file; wrong low hash; missing VAE key; tampered official uint8; unplanned decode; promotable run without pins; smoke-one; a wrong diffusers version fails the run closed unless `--accept-env-drift`, which then records the drift in the manifest. Verifier mutations cover: tensor hash; map-back; bin violation; NaN; low mutation; `reference_reads`; VAE pin.

### Changed file hashes (LF SHA256, this revision)

- `run_mri_native_batch.py`: `14598bf6ef54debbfc7b5e32b5994df17ef1e9a47589a50914b5d9b8a77ebb89`
- `test_mri_runner_and_verifier.py`: `4130806035457f2f10498b537cf41e9bf5995b82bec67d1a767cb8d9873759ec`

## Pending research-lead decisions

1. Accept `stable-diffusion-v1-5/stable-diffusion-v1-5` at a pinned revision as the asset behind the hard-coded `runwayml/stable-diffusion-v1-5`. If available, cross-check the unet/text_encoder file hashes against HF LFS OIDs.
2. **Cross-row clip rule at metric time.** The frozen RetinexFormer/SNR exports are clamped to [0,1]; PromptIR/DCTTA and this row's native case are raw. The recommendation is to clip every row to [0,1] at metric time. That is idempotent for the clamped rows and equals each official saturation. It must be locked before any reference is opened.
3. Input decoding differs by row: PIL here, cv2 for the other rows. Files with an EXIF rotation tag stop the run (see stop conditions).
4. If the VAE key report shows unexpected keys, the official behaviour ignores them. Record and accept, or review.
5. At metric time, assert low and GT sizes are equal for each pair: this runner's map-back targets the low geometry, but official paired-metrics resizes to GT size. Clip-at-metric proposal: every row clipped to [0,1] at metric time (pending research-lead confirmation).

## Stop conditions (no automatic rescue)

- Any assertion: pin, receipt set or hash, EXIF orientation not in {None, 1}, xFormers inactive, VAE missing keys or mismatch, official-uint8 identity, non-finite value, geometry, row count, or state drift.
- OOM, or peak VRAM beyond the 48 GB card.
- A nonzero exit.

No resize, tiling, precision or allocator change is allowed without a prospectively proven equivalent schedule. Memory estimate at 4K: about 21.8 GB of fp32 attention-feature buffer (25 steps × up-block self-attention, still live during VAE decode), plus several 4 GB full-resolution VAE activations, plus encoder skips. This is plausibly 40 GB or more. Images above 8.3 MP, such as SID-sRGB at about 12 MP, are at higher OOM risk. Measure with `probe_mri_synthetic.py --sizes <dataset max HxW>` before selecting a dataset.

Peer VRAM estimate (independent pass): at ~12 MP (4256×2848), ≈31.8 GB fp32 attention buffer + ≈11 GB encoder skip features + ≈2 GB weights before ≈6 GB decoder activations → OOM near-certain on the 48 GB card; at 4K, borderline. Run the probe first; no tiling or precision changes.

## Execution order once unblocked

1. `probe_mri_synthetic.py`. `--fake-vae-for-memory-only` gives peak VRAM as soon as the env and SD1.5 exist. With the real VAE it also gives repeat determinism, the prequant dtype and the key report.
2. Publish the VAE SHA256 and the SD revision.
3. `--smoke-one` on the chosen dataset, once the orchestrator authorizes it.
4. The full run with `--expected-count --expected-vae-sha256 --expected-sd-revision`.
5. `verify_mri_full.py --require-promotable --expected-output-manifest-sha256 <logged>`, plus `--first-row-tensor-sha256 <smoke>` if the probe showed bitwise repeatability.
