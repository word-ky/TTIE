# T075-A task 3: "Ours v2" design — per-image zero-shot denoising after the frozen TTT render

Status: design + development-on-test prototype on SDSD-indoor (DEVELOPMENT_ON_TEST, user decision T074-A / T075-A §3). Nothing here is a main-table row yet. Numbers: frozen T071-A metric, N=180 unless stated.

## 1. What the evidence says the component must do

- SSIM gap = amplified sensor noise (`ssim_diagnosis.md`): Ours' luminance term is as good as the baselines', its contrast-structure term is 0.17 lower; output noise ≈ low-image noise × gain (4.5×), ≈3.5× the baselines'.
- The renderer/selector cannot fix it (`exploratory_oracle/oracle_results.md`): the per-image GT-oracle step tops out at SSIM 0.665 (default) / 0.672 (tuned). A noise-aware TTT regulariser can only move along the same exposure–noise curve, so it is **not** recommended as the main component (one ablation at most).
- The noise is spatially correlated (white noise at Ours' measured σ would cost far less SSIM; best NLM h ≈ 6×255·σ_Imm; plain ZS-N2N, which assumes pixel-independent noise, removes little of it).

## 2. Proposed v2 pipeline

low x → frozen T070-A (CLIP gate → 12-D CommonRegion2 TTT, 27 Adam steps → safety/utility step selector) → rendered y → **D(y)**: OpenCV `fastNlMeansDenoisingColored` (template 7, search 21, h = hColor = κ·255·σ̂(y), σ̂ = Immerkær estimate on y itself) → output.

- Uses only the current low image and its own render; no paired data, no training set, no GT (κ is one global constant).
- Placement: after rendering (the noise that hurts SSIM is the amplified one; σ̂ is measured where the denoiser acts; the 8-bit low has σ ≈ 0.8 DN, i.e. noise at the quantisation level, which makes pre-render denoising hard). Pre-render is an ablation.
- New knob: κ (default 6 from SDSD dev; SSIM is flat for κ ∈ [4, 8], see below). Optional: NLM window sizes.
- Cost at 960×512: NLM 0.67 s/image on one CPU thread, 0 VRAM (T070-A TTT itself ≈0.78 s/image on the 4090). Learned alternative ZS-N2N: 13.3 s/image on the 4090, 2000 iterations, peak 0.72 GiB (6.9 GiB transient on the first image from cudnn.benchmark). CBM3D: ≈35 s/image on the host's 12-CPU quota (not viable for sweeps).

## 3. Prototype results (development on test, SDSD-indoor)

NLM on the frozen outputs (`v2_proto/nlm_all_rows.py`, full N=180; same filter applied to every row as a fairness control):

| Row | h=0 (as frozen) | h=4 | h=8 | h=12 | h=15 | h=18 | h=22 |
|---|---|---|---|---|---|---|---|
| retinexformer | 19.13 / 0.7874 | 19.18 / 0.8225 | 19.19 / 0.8223 | 19.19 / 0.8200 | 19.19 / 0.8179 | 19.19 / 0.8157 | 19.17 / 0.8125 |
| snr_aware | 18.05 / 0.7695 | 18.08 / 0.7954 | 18.08 / 0.7957 | 18.07 / 0.7933 | 18.07 / 0.7912 | 18.05 / 0.7889 | 18.02 / 0.7858 |
| promptir | 18.33 / 0.7644 | 18.33 / 0.7993 | 18.33 / 0.8030 | 18.33 / 0.8040 | 18.32 / 0.8040 | 18.32 / 0.8037 | 18.31 / 0.8030 |
| promptir_dctta | 18.69 / 0.7741 | 18.74 / 0.8190 | 18.75 / 0.8228 | 18.74 / 0.8220 | 18.74 / 0.8206 | 18.73 / 0.8189 | 18.71 / 0.8163 |
| mr_illuminate | 17.80 / 0.7867 | 17.80 / 0.8163 | 17.79 / 0.8142 | 17.77 / 0.8108 | 17.75 / 0.8081 | 17.74 / 0.8054 | 17.71 / 0.8021 |
| quadprior | 17.54 / 0.7895 | 17.51 / 0.8250 | 17.51 / 0.8243 | 17.49 / 0.8220 | 17.48 / 0.8200 | 17.46 / 0.8179 | 17.44 / 0.8152 |
| ours_ttt (frozen knobs) | 18.24 / 0.6408 | 18.33 / 0.7702 | 18.38 / 0.7980 | 18.40 / 0.8096 | 18.41 / 0.8155 | 18.41 / 0.8183 | 18.39 / 0.8182 |
| ours_ttt_target_tuned | 19.03 / 0.6252 | 19.18 / 0.7730 | 19.26 / 0.8040 | 19.30 / 0.8166 | 19.32 / 0.8235 | 19.33 / 0.8280 | 19.33 / 0.8294 |

Blind strength h = κ·255·σ̂ (per image, nearest grid h): frozen κ=4/6/8 → 18.41/0.811, 18.40/0.818, 18.37/0.816; tuned → 19.32/0.823, 19.33/0.829, 19.32/0.828.

Learned per-image alternative (ZS-N2N, `v2_proto/zsn2n_proto.py`, frozen Ours output, 15-image stratified subset = every 12th frame, 2–3 per video):

| Denoiser (same 15 images) | PSNR | SSIM | s/image |
|---|---|---|---|
| none | 18.04 | 0.6265 | – |
| ZS-N2N (plain) | 18.13 | 0.6902 | 13.3 |
| ZS-N2N on pixel-shuffle stride 2 | 18.19 | 0.7345 | 13.7 |
| ZS-N2N on pixel-shuffle stride 4 | 18.24 | 0.7572 | 13.6 |
| NLM h=8 / 18 | 18.19 / 18.24 | 0.7968 / 0.8205 | 0.67 (CPU) |

(30-image subset, every 6th frame: plain ZS-N2N 18.39/0.634 → 18.48/0.697 frozen; 18.97/0.615 → 19.08/0.685 tuned.)

Reading:
1. NLM removes Ours' SSIM deficit: 0.641 → 0.818 (frozen knobs), 0.625 → 0.829 (tuned). PSNR moves only +0.17 / +0.30 dB; PSNR is limited by exposure/tone error (BM-PSNR, step-oracle), not noise.
2. **Fairness control matters.** The same NLM also lifts every baseline (to 0.796–0.825 at its best h), because the SDSD GT is very clean (σ_Imm 0.0026). Against unfiltered baselines, Ours+NLM is #1 on SSIM at any h ≥ 8. Against baselines given the same filter at their own best h: tuned Ours+NLM 0.829 vs QuadPrior+NLM 0.825 (SSIM) and 19.33 vs RetinexFormer+NLM 19.19 dB (PSNR), i.e. parity-to-slightly-ahead, well inside the cluster-bootstrap noise (G=6); frozen-knob Ours+NLM (0.818 / 18.41) stays behind the best filtered baselines.
3. ZS-N2N as published is the wrong tool here (pixel-independence violated); pixel-shuffle downsampling recovers part of the gap (stride 4: +0.13 SSIM) but remains 0.06 below NLM at 20× the cost.

## 4. Ablation plan for v2 (declare before running; SDSD-indoor = development, disclosed)

1. v2 = frozen T070-A + NLM(κ fixed) vs frozen T070-A (main effect of D).
2. κ sweep {3, 4, 6, 8} and fixed-h sweep (full curves reported).
3. Placement: post-render (default) / pre-render (denoise the low, rerun TTT) / both.
4. Denoiser type at the same place: NLM / ZS-N2N PD-1/2/4 / CBM3D (subset, for runtime reasons).
5. Knobs: frozen T070-A vs T074-C target-tuned + D (the latter is a disclosed tuned row).
6. Fairness control: the same D (same κ rule) on every baseline row, reported as a supplementary table.
7. No-TTT control: Step0 or a fixed global gain to mean 0.6 + D, to show the TTT exposure part still matters.
8. Optional: selector scored on D(y_k) instead of y_k (does denoising shift the best stop later/brighter?).

## 5. Headroom from the oracle study

- PSNR: step selector gap 0.74 dB (default) / 0.79 dB (tuned); denoising adds 0.17–0.30 dB. Best non-oracle combination (tuned + NLM) 19.33.
- SSIM: step oracle ≤ 0.672; denoising gives +0.18–0.20 (to 0.818–0.829); per-image oracle strength adds only +0.002 over one global h, so a blind rule loses almost nothing.
