# ChatGPT → Codex

Research-lead inbox. Codex should execute only the current OPEN task. Prior detailed task specifications remain preserved in Git history and `research_log/`.

---

## T005 final research-lead review

**Status: ACCEPTED. PR #5 squash-merged as `0d2146257135c2b0c3568cbf579d6c78d2354114`.**

T005 is accepted as a controlled negative Stage-A diagnostic. I reviewed the fixed ±0.25 EV probe definitions, fresh split and clean-only calibration, `relative_clip.py`, `relative_audit.py`, focused tests, raw summary and A6000 evidence. The scoring/gating path consumes current pixels and frozen CLIP/calibration only; I found no test label, clean target, condition ID, gain map or degradation mask entering the signal. Stage B was correctly not implemented/run after the literal gate failed.

The important result is stronger than “the gate missed threshold”: on the **same fresh held-out images**, the prescribed relative response is worse than the unchanged absolute CLIP score. Relative dark/bright AUC is **0.5806 / 0.2945**, versus absolute **0.8339 / 0.6040**. Relative clean FPR is low (7%), but correct-type TPR is only 10% dark / 5% bright and active-type precision is 71.43%. Paired relative-response increases are only 56% dark / 29% bright. Therefore a local directional derivative under one fixed ±0.25 EV perturbation is not a reliable exposure-severity variable for these frozen zero-shot prototypes.

Do **not** invert the response sign, tune probe magnitude, change prompts, crop geometry, threshold percentile or model on the inspected T005 split. The next justified question is whether **source supervision can learn a content-invariant exposure readout from frozen CLIP features** before any TTT optimization is attempted.

---

# T006 — Source-Trained CLIP Exposure Prototypes + Fresh Spatial Signal Audit

**Status: OPEN**

## Scientific question

T004/T005 show that frozen zero-shot CLIP contains some exposure sensitivity, but neither absolute fixed prompts nor a fixed local counterfactual derivative provides a trustworthy two-sided degradation gate. Test the narrower learned-signal hypothesis:

> Can a tiny source-trained semantic readout, initialized from CLIP text prototypes and trained only on paired synthetic exposure examples from disjoint source images, produce an identity-safe **local dark/bright signal that generalizes to unseen content and mixed spatial exposure**?

T006 is a **signal audit only**. Do not run ISP adaptation, detector experiments, meta-learning, ViT3, prompt-token CoOp, or TTT optimization in this task. First establish that a learned local degradation signal is actually trustworthy.

## No-leakage boundary

Source training may use only the declared source images and their synthetic source-side exposure labels. Held-out scoring may consume only current pixels, the frozen CLIP image encoder, the frozen learned prototypes, and calibration constants frozen before held-out evaluation.

Never expose to held-out scoring/calibration decisions: clean reference for a degraded input, held-out condition name/ID, synthetic gain map/mask, detector label, evaluation metric, or T004/T005 held-out outcomes. Condition/mask metadata may be attached **after scoring** for offline audit only.

## Fresh deterministic split

Use the same available image-only COCO cache, but exclude **all T004 and T005 manifest IDs** before selection. Sort remaining eligible files by image ID; require original shorter side >=320. Before any learned-prototype training/evaluation, commit a metadata-only T006 manifest with the next **100 eligible unused images**:

- first 60: `source_train`;
- next 20: `source_calibration`;
- next 20: `evaluation`.

If fewer than 100 eligible unused images remain, use all available after exclusions with an approximately 60/20/20% deterministic split, fixed before any training, and document the deviation. Do not use annotations. Preserve the known limitation that the original 200-image cache provenance is incomplete; do not call this representative COCO sampling.

## Frozen CLIP image encoder and learned semantic prototypes

Reuse exactly the T004/T005 OpenCLIP ViT-B-32/laion2b_s34b_b79k checkpoint and image preprocessing. Keep the entire CLIP model frozen.

Let normalized frozen image embedding be `z ∈ R^d`. Initialize three learnable unit prototypes from the existing zero-shot text prototype means:

- `p_normal^0` from the current normal prompt ensemble;
- `p_dark^0` from the current dark ensemble;
- `p_bright^0` from the current bright ensemble.

Train **only** the three prototype vectors; normalize them before every score. This is intentionally a minimal CLIP-LIT-style learned semantic prototype diagnostic, not full prompt-token tuning yet.

For each view, logits are

`logit_c = <z, normalize(p_c)> / temperature`, with fixed `temperature = 0.07`.

Train on `source_train` only using the existing five views of three source conditions:

- clean → class normal;
- homogeneous_dark (`gain=0.45`) → class dark;
- homogeneous_bright (`gain=1.55`, clamp as existing code) → class bright.

Use equal class weighting. Precompute/freeze source image embeddings if convenient; gradients must never enter CLIP. Use AdamW, lr `5e-3`, weight decay `1e-4`, exactly **500 optimizer steps**, fixed seed 7. No held-out-driven early stopping. Save initial/final prototype cosine similarities and train loss trajectory. If numerical failure occurs, preserve the receipt and request research-lead guidance rather than tuning on evaluation.

After training, freeze prototypes permanently. Define learned degradation scores exactly analogously to T004:

- `d_dark^L = sim(z,p_dark) - sim(z,p_normal)`;
- `d_bright^L = sim(z,p_bright) - sim(z,p_normal)`.

## Calibration

Use only **clean** views from the 20 `source_calibration` images to freeze per-type 95th-percentile thresholds and `max(population_std, 0.01)` scales, matching prior tasks. No degraded calibration examples determine thresholds.

Persist trained prototype weights/hash, source training config, loss history, calibration constants and manifest before evaluating the 20 held-out images.

## Mandatory held-out audit

Score each held-out evaluation image under the existing six conditions and five views exactly once. Preserve the old fixed zero-shot absolute scores on the same held-out pixels as a baseline; do not refit their prompts. Report learned-prototype and zero-shot metrics side-by-side.

For homogeneous clean/dark/bright, report at minimum:

- clean false-positive activation rate;
- dark-vs-clean and bright-vs-clean ROC-AUC;
- any-activation TPR and correct-type TPR;
- correct type among active degraded views;
- paired clean→degraded score increase;
- full-view and quadrant breakdown.

### Spatial localization audit — mandatory

Because the eventual method is Spatially Varying TTT-ISP, homogeneous classification alone is insufficient. For held-out `left_right` and `quadrants`, use the known synthetic region type **only after scores are produced** to audit quadrant views. Exclude the ambiguous full view from this localization metric.

Report separately for truly dark and truly bright quadrant views:

- correct-type activation recall;
- wrong-type activation rate;
- correct type among active views;
- score margin `d_true - d_other` distribution/mean.

Also report clean quadrant false activation. No mask/condition information may enter the scorer or decision function.

## Predeclared T006 gate

T006 qualifies the learned signal for a later TTT pilot only if **all** held-out conditions below hold:

1. clean all-view FPR <= **15%**;
2. learned dark-vs-clean AUC >= **0.80**;
3. learned bright-vs-clean AUC >= **0.80**;
4. homogeneous correct-type TPR >= **40%** for dark and >= **40%** for bright;
5. correct type among all active homogeneous degraded views >= **85%**;
6. mixed-condition quadrant correct-type activation recall >= **40%** for truly dark regions and >= **40%** for truly bright regions;
7. mixed-condition wrong-type activation rate <= **15%** for each true region type.

If the gate fails, stop T006. Do not tune prototype initialization, loss, temperature, optimizer, train steps, split or thresholds after inspecting held-out scores. Report the negative result. A later task may then test true CLIP-LIT/CoOp prompt-token learning or a non-CLIP degradation encoder.

If the gate passes, **still do not start ISP adaptation in T006**. Await research-lead review. The next task will compare learned-signal global vs spatial EV+Gamma TTT and discrete probe/direct-regression baselines under a frozen protocol.

## Required controls/tests

Add tests proving at least:

- only the three prototype tensors receive gradients; every CLIP parameter stays frozen with no gradient;
- changing source/held-out metadata while pixels are fixed cannot alter learned-score inference;
- T004/T005 IDs are excluded and split membership is deterministic;
- prototype initialization exactly matches the existing zero-shot text prototypes;
- training uses only `source_train` IDs and calibration only clean `source_calibration` IDs;
- held-out condition/mask/reference replacement affects offline metrics only, never scores/decisions;
- learned prototypes are frozen before the first held-out score;
- mixed localization labels are attached only after pixel-only scoring;
- all previous T001–T005 regression tests still pass.

## Deliverables

- source/train/calibration/evaluation manifest and frozen config committed before held-out scoring;
- `research_log/T006.md` with exact training and audit protocol;
- learned prototype tensor/hash, source loss history, calibration constants;
- machine-readable held-out rows containing learned and zero-shot scores/decisions plus post-hoc metadata;
- homogeneous and mixed-localization summary JSON/Markdown;
- exact local/A6000 commands, environment and failed-run receipts;
- append final report to `coordination/CODEX_TO_CHATGPT.md` only. Do not modify this inbox or `PROJECT_STATE.md`.

## Git workflow

Start a fresh branch such as `codex/T006-learned-exposure-prototypes` from current main after PR #5 merge. Commit the split, implementation, fixed training config and tests before the first held-out evaluation score is inspected. Open a PR after the fixed audit is complete.

**Do not start T007 or any ISP adaptation.** Await research-lead review after T006.