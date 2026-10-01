# T073-E QuadPrior official source preparation

2026-09-25. The official [daooshee/QuadPrior](https://github.com/daooshee/QuadPrior) repository was cloned under project-local `.autodl/T073D-QuadPrior-official` at commit `cbdf02f2873cd7cc4652611e2431d7d10e371201`. Its README states training solely on COCO normal-light images, official `test.py` inference at 512-pixel resolution, float16 by default, and approximately 13 GB GPU memory on RTX 4090. The three checkpoint paths in the repository are zero-byte placeholders, not usable weights: `checkpoints/COCO-final.ckpt`, `checkpoints/main-epoch=00-step=7000.ckpt`, and `models/control_sd15_ini.ckpt`. The official Google Drive folder lists file IDs `1vXpAXVuWoNGS9NyKaoJbnITfyvKJbJkW`, `141BfDqsrhAx_UkMurfAvk0lGsKRhSTvi`, and `1IdTkoQnKBQns5-BNpWgv_CF40TsrdKCO` respectively. No target model run, reference read or metric.

The paid GPU host cannot reach `drive.google.com` directly (`Network is unreachable`). The local Windows host can enumerate the official folder and start a download, but the observed COCO checkpoint transfer rate was roughly 0.1 MB/s; the 1.61 GB file had reached only about 60 MB after 11 minutes. That local download was interrupted to avoid an unbounded foreground wait; the partial file remains under project-local `.autodl/T073D-weights/` for recovery. The other official files are approximately 1.17 and 5.22 GB per the folder and have not yet been downloaded. No weights have been replaced or inferred from paper results. Next: find an efficient authorized route for the exact official files, pin hashes, then source-bound low-only synthetic/target smoke. Status `PENDING_OFFICIAL_WEIGHTS`, not a terminal block or frozen output.

A task-owned parallel byte-range downloader was tried for the 1.17 GB shared VAE checkpoint after confirming the official Drive endpoint supports HTTP 206 ranges. The observed transfer did **not** improve enough with 18 concurrent 64 MB ranges and was interrupted; no complete checkpoint was produced. The partial range files and downloader remain project-local for possible resumption, not accepted model assets. The user was asked for an existing exact-weight path or a faster official/public link, without requesting credentials.

## Source-only audit, 2026-09-25 ~19:10 +08 (no weights, no execution)

Audited commit `cbdf02f2873cd7cc4652611e2431d7d10e371201` (unchanged). Inference reads only `glob(input_folder/*.*)` (`test.py:117,129`); `paired-metrics.py` and `coco_dataset.py` are not on the inference path. The three author-released files are the only weights. There are no runtime downloads; the in-repo `empty_embedding.pkl` is loaded from a cwd-relative path (`cldm.py:326`). `main-epoch=00-step=7000.ckpt` is the same file MR. Illuminate needs.

- **Official path:**
  - Resizes the short side to 512 and rounds both sides to multiples of 64, using INTER_AREA when downscaling (`annotator/util.py:28-37`, `test.py:72`). A 3840×2160 input therefore produces a **896×512** output, which is never mapped back.
  - The only official mapping to the evaluation geometry is a bilinear `cv2.resize` to the GT size in the metric script (`paired-metrics.py:74`).
  - Sampling is DPM-Solver++ order 3, 10 steps, fp16. `--use_float16` cannot be disabled from the CLI (`test.py:25`).
  - `seed_everything(0)` is called before every image (`test.py:85`), so results are independent of processing order.
  - `scale=9` guidance has no effect, because cond and uncond are identical.
- **Environment:** torch 1.12.1+cu113, PL 1.6.5, transformers 4.19.2, deepspeed (imported even at inference, `cldm.py:21`). This needs a separate env with a newer torch/CUDA for sm_89, and that version combination is unverified.
- **Research-lead decision proposed (before any QuadPrior run):**
  - Keep the official 512-short-side inference.
  - Map outputs to 3840×2160 using the official bilinear `cv2.resize` from `paired-metrics.py:74`, applied deterministically to the frozen outputs **before** reference opening.
  - Treat a native-4K QuadPrior run as unofficial.
  - Disclose the ~1.6% aspect change caused by the 896×512 rounding.
