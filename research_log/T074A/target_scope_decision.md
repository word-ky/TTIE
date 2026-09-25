# T074-A user decision: drop UHD-LL as a main-table target before any reference access

2026-09-25 ~23:00 +08. **Decision by the user (project owner), recorded before any UHD-LL reference was opened or metric computed.**

## Decision

The paper's target scope is narrowed to **extremely dark ("near-black") low-light images**, the regime of the LOL source domain. UHD-LL is removed from the main-table target set and replaced by paired real low-light benchmarks whose low images are near-black.

## Reason (low-only information only)

Under the sealed no-active abstention rule, frozen T070-A's CLIP exposure gate found no active region on 49/150 UHD-LL low images (`research_log/T073C/ttt_freeze_receipt.json`). The gate marks a region active only when CLIP judges it more under/over-exposed than the 95th percentile of clean normal-light calibration images (`ttie/clip_signal.py`). On all 100 official LOL-v2 Real test images every image executed TTT (T071-A selected step ≥ 17). The user judges UHD-LL's moderately dark 4K night scenes to be outside the method's intended operating regime. This reason uses target-low structure only; no UHD-LL reference, PSNR/SSIM or cross-method outcome informed it.

## Integrity rules adopted with the decision

1. **UHD-LL references stay sealed permanently.** No UHD-LL clean/reference payload may be opened and no UHD-LL metric computed for any method, so the drop cannot depend on UHD-LL outcomes.
2. All UHD-LL evidence is retained unchanged: the six frozen rows (RetinexFormer, SNR-Aware, PromptIR, PromptIR+DCTTA, Ours-Step0, Ours-TTT), receipts, verifiers and logs. Nothing is deleted or rewritten.
3. Replacement targets are chosen by **method-independent** criteria only: paired real capture, canonical test split, availability, no LOL-v2 Real overlap, and a prospectively fixed low-image darkness criterion (e.g. mean luma). **Gate activation rate or any method's behaviour/performance on candidate data must not be used for selection.**
4. The darkness criterion and the chosen targets are sealed in a new preregistration before any method runs on them. Method roles, source checkpoints (LOL-v2-Real-source-frozen), metric implementation, bootstrap seed `20260922`/10,000 resamples and the no-active abstention rule carry over unchanged.
5. The paper must disclose that UHD-LL was a preregistered target dropped before reference opening, with this reason and the 49/150 abstention count, and must scope its claims to the near-black regime.

Counters at decision time: UHD-LL `reference_reads=0`, `metrics=0`.
