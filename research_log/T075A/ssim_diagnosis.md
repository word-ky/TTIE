# T075-A task 1: SSIM diagnosis, SDSD-indoor (N=180, 6 videos x 30 frames)

Analysis only, existing frozen outputs (8 gated rows + T074-C target-tuned Ours row). GT read only via `metrics.Context.reference` (gate receipt `a21eb87d…`); outputs via `metrics.load_output` (hash-checked); PSNR/SSIM = frozen T071-A `core.metrics`. Script `ssim_diag.py`, per-image records `ssim_diag.json`.

Definitions. BM = output scaled by mean(GT)/mean(output) (one scalar per image), clipped. CM = per-channel scaling (colour + brightness matched). SSIM terms use the frozen filter (Gaussian σ=1.5, truncate 3.5, reflect, C1=.01², C2=.03²): SSIM = mean(l · cs), l = luminance term, cs = contrast-structure term, split into c (contrast) and s (structure, C3=C2/2). σ_Imm = Immerkær (1996) noise estimate (whole image, RGB mean, [0,1] units). σ_hp = 1.4826·MAD of (img − Gauss₁(img)) inside GT-flat pixels (lowest 30% blurred-GT gradient), on the BM output.

## Conclusion

Ours loses SSIM through **noise amplification**, not brightness, colour or large-scale structure.

- The luminance term of Ours is as good as or better than every baseline (0.923 frozen / 0.936 tuned vs 0.892–0.924). The contrast-structure term is where the gap is: 0.687 / 0.657 vs 0.831–0.862. That term alone accounts for the full ≈0.15 SSIM gap (mean l gap vs RetinexFormer +0.006, mean cs gap −0.166).
- Brightness matching does not close the gap: BM-SSIM of Ours is 0.618 (frozen) / 0.632 (tuned) vs 0.769–0.798 for baselines. Colour matching adds nothing on top (CM-SSIM = BM-SSIM to 3 decimals for every row). BM-PSNR gains are similar for all rows (+1.0 to +2.1 dB), so the PSNR side is mostly exposure error shared by all methods.
- Output noise of Ours is ≈3.5× the baselines' and ≈5× the GT's (σ_Imm 0.0126 / 0.0140 vs 0.0031–0.0044, GT 0.0026; flat-region σ_hp 0.012 vs 0.004–0.006, GT 0.0018). It equals the low image's noise times the applied gain (Step0 σ_Imm × mean-gain 4.48 predicts 0.0132; measured 0.0126): the 12-D EV/gamma/gain renderer cannot remove noise, while every baseline is a trained restoration network that denoises.
- The pattern holds in all 6 videos: Ours' cs term is 0.12–0.21 below every baseline in every video, while its l term is at least as high as the lowest baseline's in 5 of 6 videos (pair4: −0.001). The tuned row trades SSIM for PSNR because its brighter exposure target (0.7) amplifies noise more (σ_Imm 0.0140).
- Consequence: the SSIM gap cannot be fixed inside the current 12-D renderer or by step/knob selection (step-oracle numbers in `exploratory_oracle/`); a noise-removal component is required.

## Per-row means

| Row | PSNR | SSIM | BM-PSNR | BM-SSIM | CM-SSIM | l | c·s | c | s | σ_Imm | σ_hp flat (BM) | mean out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| retinexformer | 19.13 | 0.7874 | 20.33 | 0.7976 | 0.7980 | 0.9169 | 0.8535 | 0.9537 | 0.8887 | 0.0037 | 0.0041 | 0.499 |
| snr_aware | 18.05 | 0.7695 | 19.80 | 0.7819 | 0.7820 | 0.8918 | 0.8615 | 0.9565 | 0.8939 | 0.0044 | 0.0048 | 0.406 |
| promptir | 18.33 | 0.7644 | 20.10 | 0.7687 | 0.7686 | 0.9080 | 0.8337 | 0.9424 | 0.8776 | 0.0032 | 0.0044 | 0.441 |
| promptir_dctta | 18.69 | 0.7741 | 20.36 | 0.7828 | 0.7828 | 0.9237 | 0.8306 | 0.9418 | 0.8755 | 0.0034 | 0.0044 | 0.482 |
| mr_illuminate | 17.80 | 0.7867 | 19.67 | 0.7835 | 0.7836 | 0.9119 | 0.8602 | 0.9560 | 0.8927 | 0.0031 | 0.0048 | 0.415 |
| quadprior | 17.54 | 0.7895 | 19.65 | 0.7819 | 0.7825 | 0.9240 | 0.8517 | 0.9489 | 0.8906 | 0.0033 | 0.0058 | 0.394 |
| ours_step0 | 7.40 | 0.3189 | 16.55 | 0.5863 | 0.5862 | 0.3685 | 0.8461 | 0.9046 | 0.9216 | 0.0030 | 0.0124 | 0.095 |
| ours_ttt | 18.24 | 0.6408 | 19.26 | 0.6180 | 0.6180 | 0.9229 | 0.6871 | 0.8466 | 0.8068 | 0.0126 | 0.0123 | 0.417 |
| ours_ttt_target_tuned | 19.03 | 0.6252 | 19.85 | 0.6315 | 0.6315 | 0.9363 | 0.6572 | 0.8221 | 0.7951 | 0.0140 | 0.0121 | 0.478 |
| GT | | | | | | | | | | 0.0026 | 0.0018 | 0.474 |

## Per-video (30 frames each)

| Row / metric | pair1 | pair11 | pair19 | pair21 | pair4 | pair9 |
|---|---|---|---|---|---|---|
| retinexformer PSNR | 18.3180 | 19.2665 | 21.6710 | 22.5272 | 17.5456 | 15.4524 |
| snr_aware PSNR | 19.1572 | 11.0965 | 22.3666 | 23.6768 | 17.5698 | 14.4444 |
| promptir PSNR | 19.7766 | 19.0913 | 16.9003 | 22.5649 | 15.0440 | 16.5903 |
| promptir_dctta PSNR | 19.0640 | 19.3814 | 18.8525 | 23.1002 | 16.4651 | 15.2910 |
| mr_illuminate PSNR | 18.1406 | 15.9340 | 18.6914 | 21.8524 | 14.9378 | 17.2294 |
| quadprior PSNR | 16.6188 | 16.7718 | 18.3591 | 22.8939 | 12.9771 | 17.5903 |
| ours_ttt PSNR | 17.6781 | 17.8652 | 18.6501 | 23.8741 | 13.3016 | 18.0619 |
| ours_ttt_target_tuned PSNR | 18.7407 | 18.0250 | 20.6106 | 23.5922 | 16.5087 | 16.7278 |
| retinexformer SSIM | 0.7271 | 0.8305 | 0.8187 | 0.7832 | 0.7402 | 0.8248 |
| snr_aware SSIM | 0.7497 | 0.7004 | 0.8148 | 0.7841 | 0.7435 | 0.8247 |
| promptir SSIM | 0.7697 | 0.8352 | 0.7301 | 0.7458 | 0.6796 | 0.8262 |
| promptir_dctta SSIM | 0.7180 | 0.8340 | 0.7808 | 0.7678 | 0.7306 | 0.8133 |
| mr_illuminate SSIM | 0.7493 | 0.8089 | 0.8004 | 0.7670 | 0.7356 | 0.8588 |
| quadprior SSIM | 0.7492 | 0.8093 | 0.7977 | 0.8136 | 0.7123 | 0.8549 |
| ours_ttt SSIM | 0.5989 | 0.6618 | 0.6617 | 0.6666 | 0.5343 | 0.7217 |
| ours_ttt_target_tuned SSIM | 0.5776 | 0.6381 | 0.6556 | 0.6535 | 0.5527 | 0.6738 |
| retinexformer BM-SSIM | 0.7441 | 0.8213 | 0.8147 | 0.7977 | 0.7435 | 0.8646 |
| snr_aware BM-SSIM | 0.7490 | 0.7601 | 0.8127 | 0.7941 | 0.7401 | 0.8355 |
| promptir BM-SSIM | 0.7677 | 0.8363 | 0.7392 | 0.7441 | 0.6745 | 0.8503 |
| promptir_dctta BM-SSIM | 0.7331 | 0.8379 | 0.7744 | 0.7745 | 0.7174 | 0.8593 |
| mr_illuminate BM-SSIM | 0.7425 | 0.8034 | 0.7978 | 0.7788 | 0.7270 | 0.8515 |
| quadprior BM-SSIM | 0.7378 | 0.7991 | 0.7979 | 0.8098 | 0.6996 | 0.8468 |
| ours_ttt BM-SSIM | 0.5668 | 0.6367 | 0.6266 | 0.6672 | 0.5066 | 0.7041 |
| ours_ttt_target_tuned BM-SSIM | 0.5705 | 0.6479 | 0.6330 | 0.6755 | 0.5497 | 0.7123 |
| retinexformer l | 0.8688 | 0.9421 | 0.9539 | 0.9179 | 0.9095 | 0.9092 |
| snr_aware l | 0.8856 | 0.7726 | 0.9509 | 0.9254 | 0.9176 | 0.8984 |
| promptir l | 0.9302 | 0.9494 | 0.8762 | 0.9058 | 0.8562 | 0.9302 |
| promptir_dctta l | 0.8800 | 0.9465 | 0.9466 | 0.9328 | 0.9193 | 0.9172 |
| mr_illuminate l | 0.8855 | 0.9038 | 0.9483 | 0.8972 | 0.9013 | 0.9352 |
| quadprior l | 0.8907 | 0.9175 | 0.9507 | 0.9538 | 0.8853 | 0.9461 |
| ours_ttt l | 0.9014 | 0.9326 | 0.9576 | 0.9455 | 0.8548 | 0.9457 |
| ours_ttt_target_tuned l | 0.9026 | 0.9380 | 0.9646 | 0.9424 | 0.9263 | 0.9441 |
| retinexformer c·s | 0.8276 | 0.8786 | 0.8565 | 0.8521 | 0.8029 | 0.9032 |
| snr_aware c·s | 0.8412 | 0.9073 | 0.8551 | 0.8468 | 0.8007 | 0.9177 |
| promptir c·s | 0.8173 | 0.8749 | 0.8242 | 0.8214 | 0.7804 | 0.8840 |
| promptir_dctta c·s | 0.8019 | 0.8754 | 0.8208 | 0.8190 | 0.7859 | 0.8805 |
| mr_illuminate c·s | 0.8439 | 0.8931 | 0.8433 | 0.8525 | 0.8113 | 0.9168 |
| quadprior c·s | 0.8378 | 0.8784 | 0.8377 | 0.8528 | 0.8014 | 0.9024 |
| ours_ttt c·s | 0.6494 | 0.7003 | 0.6875 | 0.6992 | 0.6267 | 0.7596 |
| ours_ttt_target_tuned c·s | 0.6211 | 0.6679 | 0.6752 | 0.6858 | 0.5843 | 0.7089 |
| retinexformer σ_hp | 0.0034 | 0.0063 | 0.0040 | 0.0036 | 0.0043 | 0.0030 |
| snr_aware σ_hp | 0.0035 | 0.0084 | 0.0036 | 0.0036 | 0.0043 | 0.0054 |
| promptir σ_hp | 0.0041 | 0.0044 | 0.0054 | 0.0044 | 0.0048 | 0.0033 |
| promptir_dctta σ_hp | 0.0041 | 0.0043 | 0.0052 | 0.0043 | 0.0053 | 0.0034 |
| mr_illuminate σ_hp | 0.0045 | 0.0063 | 0.0049 | 0.0037 | 0.0051 | 0.0042 |
| quadprior σ_hp | 0.0057 | 0.0062 | 0.0063 | 0.0047 | 0.0067 | 0.0052 |
| ours_ttt σ_hp | 0.0140 | 0.0133 | 0.0136 | 0.0116 | 0.0103 | 0.0109 |
| ours_ttt_target_tuned σ_hp | 0.0137 | 0.0129 | 0.0129 | 0.0113 | 0.0116 | 0.0104 |
