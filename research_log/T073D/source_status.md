# T073-D MR. Illuminate official source preparation

2026-09-25. The official author repository is [joshua-cho-story/MR-Illuminate](https://github.com/joshua-cho-story/MR-Illuminate), advertised by its README as released 2026-08-10; observed HEAD commit `013e013e0ab2112dfa5d43f2dc87d34aee4ee328`. The official README specifies `main.py` with custom `--img_dir_path`, `--save_dir`, `--scaled_save_dir`, and requires the exact QuadPrior bypass VAE checkpoint `main-epoch=00-step=7000.ckpt`. The repository directs that checkpoint to be downloaded from QuadPrior's official Google Drive folder; the file is not in this project or on the paid GPU host. No target model run, reference read or metric. Do not substitute another VAE or a different inference method.

The author repository's custom inference path appears low-only at the CLI level, but the source and imports still require audit before target execution. The shared VAE artifact download is a genuine dependency for both MR. Illuminate and QuadPrior. Next: obtain/hash that official checkpoint, pin full source files and defaults, test a synthetic image, then execute one canonical low-only smoke. Status `PENDING_CHECKPOINT_AND_EXECUTION_GATE`, not `FROZEN_OUTPUTS`.

## Source-only audit, 2026-09-25 ~19:10 +08 (no weights, no execution)

Audited HEAD `013e013e0ab2112dfa5d43f2dc87d34aee4ee328` (unchanged). Inference reads only `--img_dir_path` (`main.py:244,284,580`); ground truth is touched only by `evaluation/*`, which the UHD-LL runner must not call.

- **Assets:** the author-released QuadPrior VAE `main-epoch=00-step=7000.ckpt`, of which only the `my_vae.` keys are loaded, with `strict=False` (`main.py:182-204`; the missing/unexpected key report must be recorded). Plus the third-party `runwayml/stable-diffusion-v1-5` diffusers pipeline, hard-coded at `main.py:41`. That repo id now redirects to `stable-diffusion-v1-5/stable-diffusion-v1-5` (observed revision `451f4fe`). The unet and text encoder are used; the SD VAE and safety checker are downloaded but not used. About 5.5 GB. It must be pinned by revision and file hashes.
- **Official path:** native resolution, with each side rounded to a multiple of 8 (LANCZOS) and resized back bilinearly (`main.py:291-299,532`). A 3840×2160 input is processed at native size with no resampling. Inputs whose mean brightness is below 30 are scaled globally to mean 30 (`main.py:304`). Sampling is DDIM, 25 steps, inversion + self-attention injection, empty prompt, under `autocast(float32)` (`main.py:147`). `--scaled_save_dir` writes a brightness-scaled baseline, which is not the method output.
- **Determinism:** `seed_everything(1)`; DDIM eta=0; the AdaIN noise generator is seeded with 145, with shape `[1,4,64,64]` (`main.py:456,485`).
- **Environment:** Python 3.9, torch 2.0.1, xformers 0.0.20 (required: `main_utils.py:10,73,157`), diffusers 0.17.1, transformers 4.30.2. This needs a separate conda env on the host.
- **Risks:**
  - Per-image exceptions are caught and skipped (`main.py:614`), so the runner must assert exactly 150 outputs.
  - The native-4K VRAM estimate is roughly 35–50 GB against the 48 GB RTX 4090. That is unmeasured and may OOM. If it does OOM, any rescue needs a prospectively proven execution-equivalent schedule, like PromptIR's. Resizing is not allowed.
- **Research-lead decision proposed:** accept the redirected official SD1.5 repository at a pinned revision as the same third-party asset.
