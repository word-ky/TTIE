# Per-dataset Ours development "ceiling" rows (user decision, 2026-09-26 evening, +08)

The user requires, on **every** target dataset, an Ours result developed (tuned) on that dataset's own test GT, reported as a **ceiling reference**, per `research_log/T074A/ours_target_tuning_decision.md`.

- Row id `ours_v2_ceiling_<target>`: Ours-TTT knobs tuned on the target's test GT with the same peer-reviewed procedure as SDSD-indoor (`ours_tuning.py`: round-1 grid `tuning_grid_round1.json`, round-2 rule, selection = highest mean PSNR, ties within 0.01 dB by SSIM, then normalised distance), followed by the v2 denoiser D with κ selected from {4, 5, 6, 7, 8} by the same rule on the tuned outputs. Every setting and κ is logged.
- Run only **after** the target's reference gate is open, i.e. after all its non-tuned and v2 rows are frozen; frozen rows are never rerun, so held-out v2 evidence on LSRW/SMID/SID is unaffected.
- Labelled everywhere `tuned_on_test_gt: true` / "ceiling, not held-out". The existing SDSD-indoor tuned row plays this role for SDSD-indoor (plus κ selection).
- Order: LSRW now; SMID after its gate; SID after its gate.
