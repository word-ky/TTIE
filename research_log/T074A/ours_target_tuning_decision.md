# T074-A addendum: user decision on Ours development on the target test data

2026-09-25 (late evening, +08). **Decision by the user (project owner).**

## Decision

For the new near-black targets, Ours (the per-image TTT method) may be developed on the target domain and evaluated on the same target test images: its thresholds and hyperparameters may be chosen using the target test images **including their GT/reference and quality metrics**, and results are reported on those same images. The user will state this explicitly in the paper. This supersedes, for the new targets, the frozen-T070-A / no-target-tuning rule of the Phase-1 program. It does not change the dropped UHD-LL target (its references stay sealed permanently).

Assistant's recorded note (for transparency): selecting hyperparameters with test-set GT and reporting on the same images yields optimistic estimates relative to held-out evaluation; the paper must disclose it.

## Safeguards adopted with the decision

1. **Baselines first.** All seven non-tuned rows (RetinexFormer, SNR-Aware, PromptIR, PromptIR+DCTTA, MR. Illuminate, QuadPrior, and the pre-tuning Ours-Step0/Ours-TTT with frozen T070-A settings) are run on the target low images and their outputs frozen, hashed and verified **before any target GT is opened**. Their outputs are never rerun after GT access.
2. The frozen-T070-A Ours rows are kept as their own rows, so the paper can show frozen vs target-tuned Ours.
3. Every Ours development run after GT access is logged (settings tried, metric per setting, selection rule), so the tuning search is fully reported.
4. Optional robustness check offered to the user: scene-disjoint two-fold split of the target test set (tune on one half, evaluate on the other, swap), reported alongside.
5. Baseline rows keep LOL-v2-Real source checkpoints; if target-trained official baseline checkpoints are added as reference rows, that is a separate, disclosed decision.
