# EXPLORATORY_ORACLE_NOT_FOR_PAPER — T075-A task 2: what would make Ours rank first (SDSD-indoor)

**Label: EXPLORATORY_ORACLE_NOT_FOR_PAPER.** Every number here selects something per image or per dataset with the SDSD-indoor GT. None of it is a main-table row or may be merged into one (user decision 2, `../plan_and_amendments.md`).

Frozen reference rows (T074-C, `research_log/T074C/SDSD_indoor/metrics`): RetinexFormer 19.130 / 0.7874 (best PSNR), QuadPrior 17.535 / 0.7895 (best SSIM); SSIM range of all six baselines 0.764–0.790. Ours-TTT frozen 18.238 / 0.6408; target-tuned 19.034 / 0.6252.

## (a, b) Per-image GT-oracle stop step (28 states of the frozen trajectory)

`oracle_steps.py` reran frozen T070-A (default knobs and T074-C tuned knobs) through the T074-C derivation machinery; the selector reproduced the frozen / tuned row tensors bit-for-bit on all 360 image runs (hash check inside the script).

| Setting | Frozen selector | Oracle step by PSNR (PSNR / SSIM) | Oracle step by SSIM (PSNR / SSIM) | Best single fixed step (PSNR) | Median step sel / PSNR-oracle / SSIM-oracle |
|---|---|---|---|---|---|
| default | 18.238 / 0.6408 | 18.980 / 0.6237 | 15.029 / 0.6652 | step 23: 18.144 / 0.6182 | 20.5 / 23 / 15 |
| tuned | 19.034 / 0.6252 | 19.826 / 0.6304 | 15.404 / 0.6720 | step 22: 18.532 / 0.6257 | 22.5 / 22 / 15 |

Reading: the step selector leaves 0.74 dB (default) / 0.79 dB (tuned) of PSNR on the table; with tuned knobs the PSNR oracle (19.83) would beat RetinexFormer (19.13). No step choice can fix SSIM: the SSIM-oracle ceiling is 0.665 / 0.672, far below every baseline (≥0.764), and it is reached only at dark early steps (median step 15, PSNR ≈15 dB).

## (c) Ours output + OpenCV NLM (`fastNlMeansDenoisingColored`, h = hColor, 7/21), strength chosen with GT

`oracle_denoise.py`. Dataset-oracle = one h for all 180 images; per-image oracle = best h per image (grid h ∈ {2,4,6,8,10,12,15,18,22,27}). CBM3D was dropped (bm3d 4.0.3 needs ~35 s/image on the 12-CPU quota of the host).

| Input | Unfiltered | Dataset-oracle h by PSNR | Dataset-oracle h by SSIM | Per-image oracle by PSNR | Per-image oracle by SSIM | SSIM at h=8 (PSNR) |
|---|---|---|---|---|---|---|
| ours_ttt | 18.238 / 0.6408 | h=15: 18.411 / 0.8155 | h=18: 18.409 / 0.8183 | 18.441 / 0.8148 | 18.429 / 0.8207 | 0.7980 (18.383) |
| ours_ttt_target_tuned | 19.034 / 0.6252 | h=22: 19.332 / 0.8294 | h=22: 19.332 / 0.8294 | 19.378 / 0.8293 | 19.366 / 0.8317 | 0.8040 (19.256) |
| default_psnr_oracle_step | 18.980 / 0.6237 | h=18: 19.272 / 0.8174 | h=22: 19.268 / 0.8188 | 19.306 / 0.8198 | 19.298 / 0.8210 | 0.7944 (19.204) |
| tuned_psnr_oracle_step | 19.826 / 0.6304 | h=18: 20.199 / 0.8301 | h=22: 20.198 / 0.8314 | 20.245 / 0.8331 | 20.238 / 0.8338 | 0.8067 (20.105) |

Counterfactual (white Gaussian noise added to GT, frozen metric): σ=0.005: 46.03 dB / 0.9794, σ=0.01: 40.02 dB / 0.9236, σ=0.0126: 38.02 dB / 0.8851, σ=0.015: 36.52 dB / 0.8463, σ=0.02: 34.04 dB / 0.7614. Ours' measured σ_Imm is 0.0126, yet its SSIM (0.64) is far below the 0.885 of white noise at that level: its noise is spatially correlated (resize/demosaic/compression, then gain), which the Laplacian estimator under-reads and which hurts SSIM more; consistent with the best NLM h ≈ 6×(255·σ_Imm).

## Which scenario makes Ours rank #1 (vs the six frozen baselines)

- **SSIM #1**: any NLM strength h ≥ 8 on the frozen Ours output (0.798 at h=8, 0.816–0.818 at h=15–18) beats the best baseline (0.7895). The SSIM curve is flat for h ∈ [12, 27], so this does not hinge on GT-picked strength; a blind rule h = κ·255·σ_Imm(output) with κ ∈ [4, 8] gives 0.811–0.818 (frozen) / 0.823–0.829 (tuned). Step oracle alone: never.
- **PSNR #1**: needs the target-tuned knobs. Tuned + NLM (any h ≥ 8: 19.26–19.33) or tuned + oracle step (19.83) beat RetinexFormer 19.13 (margin 0.1–0.2 dB with NLM, inside the cluster-CI width). Frozen knobs: only oracle step + NLM (19.27) crosses 19.13; frozen + NLM stays at 18.41 (3rd).
- **Both #1**: target-tuned Ours + NLM (h 8–22): PSNR 19.26–19.33, SSIM 0.804–0.829, against the unfiltered baseline rows. **Caveat (fairness control, `../v2_design.md` §3):** the same NLM lifts every baseline to SSIM 0.796–0.825 at its best h; against filtered baselines tuned Ours+NLM is only at parity (0.829 vs 0.825 SSIM, 19.33 vs 19.19 dB), and frozen-knob Ours+NLM (0.818 / 18.41) is behind.
