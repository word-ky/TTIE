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

## Pinned after independent peer review (before any target GT is decoded)

Peer review found and the author fixed: multi-round search support with the round-2 rule implemented in code (`--round2`, rule exactly as below), gate requires passing verifications, `updates ≤ 27` and fixed `tau` enforced, default-reproduction check at every start, `-O` refused, tuned-row metrics flagged `tuned_on_test_gt: true`. Suite 46/46 (host CPU).

Round-2 rule (implemented in `ours_tuning.round2_grid`): per round-1 knob, gain = best COMPLETE mean PSNR (default included) − default mean PSNR; choose up to 3 knobs with strictly positive gain (ties: SSIM of best setting, then knob name); each chosen knob's top-2 values (ties: SSIM, then normalised distance to default); full product, other knobs default, `loss_weights` as one vector value; FAILED settings never counted; empty if no positive gain. Final selection over the union of all rounds.

LF SHA256: `reference_gate.py` `0337576c8f406f5f5ae2ba75966b434ddc87189fe7a8dbdb58bbb70f146db301`; `metrics.py` `1d79de3243605bd9fc3e437014cdadd8120a6056917fd204e58d360b33d61154`; `ours_tuning.py` `6ed1008bb93d54b761430c308fda11b4c2ef1beaea3f68306510a554fc370b39`; `test_t074c.py` `94b5f1429cec5ec7336a9d30a9214df12b7dc5a6d7bb0d57d9cd5e3a3e5c2a61`; `tuning_grid_round1.json` `35cf3c635ee6b3b90da211079525ae5d463eeda5f64abeb4f99351a1977fd1a3` (23 settings).
