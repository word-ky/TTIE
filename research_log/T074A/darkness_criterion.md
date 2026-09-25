# T074-A prospective near-black darkness criterion

2026-09-25 ~23:20 +08. Sealed **before any candidate target dataset was downloaded or measured** and before any method ran on any candidate.

For a candidate paired real low-light test split, compute on each **low** image only (never GT, never any model output), exactly as the canonical loader delivers it as sRGB RGB in [0,1]:

`Y = mean over pixels of (0.299 R + 0.587 G + 0.114 B)`

A candidate **passes** if both hold:

1. median of per-image `Y` ≤ 0.10;
2. at least 80% of its images have `Y` ≤ 0.15.

The measured median, 80th/90th percentiles and pass/fail are reported for every candidate measured, including failures. If the LOL-v2 Real test low images are measurable under the same code, their statistics are reported as an anchor but do not change the thresholds above.

Candidates, in the order they will be measured if available: SID-sRGB (SNR-Aware/Retinexformer preparation), SMID, LSRW (Huawei + Nikon), SDSD-indoor, SDSD-outdoor. Excluded without measurement: LOL-v1 (overlap with LOL-v2 Real), LOL-v2 Synthetic and VE-LOL-L (synthetic / likely overlap), LOL-Blur, MIT-Adobe FiveK, BVI-RLV (amplified gain).

Selection among passing candidates uses only: canonical official test split, pairing, availability, independence of scenes, and a perceptual-hash overlap check against LOL-v2 Real training. Gate activation, abstention rate, or any method's output/performance is never used. Only LOL-v2-Real-source checkpoints are used for every baseline (dataset-specific SID/SMID/SDSD checkpoints shipped by SNR-Aware/Retinexformer are forbidden).
