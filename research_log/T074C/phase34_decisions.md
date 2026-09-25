# T074-C phase 3/4 decisions (fixed before any target GT is decoded)

2026-09-26 ~03:30 +08, by the assistant under the user's full delegation; the user may override before phase 3 starts.

- Reference-opaque manifest for SDSD-indoor committed (`research_log/T074B/targets/SDSD_indoor/reference_opaque_manifest.json`, GT byte hashes only, SHA256 `dd7415dd7a8f212eeed8d7adf23e2948c0687937c63a5384f634ead655d722a2`).
- Cluster bootstrap cluster order: sorted cluster-id order (protocol §5). Low-power flag: G ≤ 10 (SDSD-indoor G = 6).
- Tuning selection: highest image-mean PSNR; ties within 0.01 dB broken by mean SSIM, then by L1 distance from defaults with each knob normalised by its range over the declared grid.
- Tuning search space (Ours-TTT target-tuned row), run with `research_log/T074C/ours_tuning.py`:
  - Round 1, one-factor-at-a-time around the frozen defaults (default setting first; it must reproduce the frozen Ours-TTT tensors bitwise):
    - `q_joint` ∈ {0.35266535990213066 (p80), 0.9747123807094724 (p90), 1.053775168916056 (p95, default)}
    - `probability_threshold` ∈ {0.3, 0.5, 0.7}
    - `lambda_value` ∈ the T067-B grid values including both endpoints and 0.875
    - `updates` ∈ {9, 18, 27} (no value above 27: the frozen step-fraction feature is defined for k ≤ 27)
    - `lr` ∈ {0.015, 0.03, 0.06}
    - `exposure_target` ∈ {0.5, 0.6, 0.7}
    - `loss_weights` ∈ {[1,10,5], [1,5,5], [1,20,5], [1,10,2.5], [1,10,10]}
  - `tau` is not tuned (its calibration maxima would become inconsistent).
  - Round 2: the three knobs whose round-1 best value changed mean PSNR the most; full product of each knob's top-2 round-1 values (8 settings).
  - Every setting (including failures) is logged in the hash-chained JSONL; the full search is reported in the paper appendix.
- The optional scene-disjoint two-fold check is not run for SDSD-indoor (only 6 clusters); decided for SID when/if it passes.
