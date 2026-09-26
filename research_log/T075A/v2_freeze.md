# Ours v2 — frozen specification (2026-09-26 afternoon, +08)

Decided by the assistant under the user's full delegation, from the SDSD-indoor development evidence in `ssim_diagnosis.md`, `exploratory_oracle/` and `v2_design.md` (SDSD-indoor = development target; its GT was open). **Frozen before any LSRW, SMID or SID GT is opened**, so LSRW/SMID/SID are held-out for v2.

- Diagnosis: Ours' SSIM gap is entirely in the contrast–structure term; output noise ≈ low-image noise × applied gain (Immerkær σ 0.0126 vs baselines 0.003–0.004). Step selection cannot fix it (GT-oracle step SSIM ceiling 0.672).
- **Ours v2 = Ours-TTT → per-image post-render denoiser D**, D = OpenCV `fastNlMeansDenoisingColored(u8, None, h, h, 7, 21)` with `h = hColor = κ · 255 · σ̂`, σ̂ = Immerkær noise estimate of the rendered image, **κ = 6**. Uses only the current image; no training.
- Two v2 rows per target: `ours_v2` (frozen T070-A default knobs + D) and `ours_v2_sdsd_knobs` (the SDSD-selected knobs q_joint = p80 0.35266535990213066, exposure_target 0.7 + D). Both fixed now.
- Fairness control rows: `<baseline>_plus_D` for each of the 7 other rows' frozen outputs, with the identical D and κ.
- Reporting: held-out claims only on targets whose GT opens after these rows are frozen (LSRW, SMID, SID). SDSD-indoor v2 numbers are development numbers. The claim is parity with trained restoration networks, not superiority; baseline+D controls are always shown.
- Ablations (development, SDSD-indoor): D on/off, κ sweep, pre- vs post-render, NLM vs pixel-shuffle ZS-N2N, frozen vs tuned knobs, no-TTT (gain + D).

## Amendment (2026-09-26 ~15:30 +08, before any SMID GT access)

On SMID the frozen T070-A selector raised an AssertionError under the SDSD-selected knobs (first at `0051__0001.png`), the same failure mode as the `lr = 0.06` setting during SDSD tuning. Declared rule for `ours_ttt_sdsd_knobs` (and therefore `ours_v2_sdsd_knobs`) on every target: **if the frozen code rejects an image under these knobs, that image's output is the frozen default-knob `ours_ttt` output**, recorded with `decision_status = KNOBS_REJECTED_FALLBACK_DEFAULT`; the count is reported with the row. SDSD-indoor and LSRW had 0 such images, so their frozen rows are unaffected. The SMID gate registry includes all 18 rows once these are produced under this rule.
