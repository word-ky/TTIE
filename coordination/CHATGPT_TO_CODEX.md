# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task. Prior specifications and evidence remain preserved in Git history and `research_log/`.

---

# Research-lead review — T023-A accepted negative; do not scale the failed 16-pair pilot, pivot one cycle to SOTA-comparison convergence

I reviewed PR #46, including the deterministic 16/589 source selection, `real_source_sobolev.py`, the EV2-matched derivative-chain test, the unchanged-validation-runner AST equivalence test, target mutation/withholding isolation, A6000 source/validation receipts, frozen artifacts, and independent metric verification. T023-A is accepted as a **leakage-safe validation-only negative/insufficient pilot** and PR #46 has been squash-merged as `9d6edcebbf8f62ae3b6b0784fffd71c42e838ffe`.

The predeclared joint gate failed. Relative to accepted T022-C (`10.2295540 dB / 0.3282315 SSIM`), the new 16-pair real-source Sobolev head reaches `10.5166353 dB / 0.3144893 SSIM`: mean PSNR improves only `+0.2870812 dB` (< `+0.50 dB` required), while mean RGB-SSIM drops `-0.0137422`. The checkpoint must not replace T022-C.

The result is scientifically useful because the source fit itself is extremely strong: on the 656 source states, positive gradient-cosine fraction becomes `1.0`, median cosine `0.9881`, direction loss `0.0168`, and log-MSE fitting error collapses. Yet this near-perfect source value/gradient fit does not transfer into the required joint PSNR/SSIM validation gain. Therefore simply scaling the same tiny-source recipe is **not justified by this pilot**; this is evidence of source/generalization mismatch, not evidence that more epochs or another seed are needed.

The information boundary is clean. Each source low-only trajectory was frozen before its paired source normal image was opened for offline supervision; the new energy checkpoint was frozen before validation inference; all 100 validation low-light outputs/decisions/trajectories were frozen before validation normal references were deployed; changing/withholding validation targets leaves inference hashes unchanged. Official LOL-v2 Real test remains untouched.

T022-C remains the best current LOL-v2 validation candidate. Because the user explicitly asked us to start converging toward a complete benchmark and strong SOTA-related comparisons, the next cycle is not another method patch. It freezes the comparison protocol and baseline roster so later official-test evaluation cannot drift toward favorable or target-assisted conventions.

---

# OPEN one-hour task — T024-A: LOL-v2 strong-baseline protocol and fair-comparison freeze

**Work budget: about one hour. One engineering objective only: freeze a paper-ready, target-free LOL-v2 Real baseline/comparison protocol before any official-test run, with exact provenance and fairness classification for a small set of strong published baselines.**

## Hypothesis / engineering objective

The current bottleneck to paper-level convergence is no longer lack of another small TTT tweak; it is that LLIE papers and repositories often use incompatible training splits, preprocessing, metric implementations, and in some cases target-assisted test normalization. Before running a final benchmark, bind a fixed baseline roster and determine exactly which comparison recipes are admissible under our rule that inference must never use normal-light targets or evaluation metrics.

Audit exactly these five baseline families, using only their official paper/repository/project sources when available:

1. **Retinexformer** (ICCV 2023),
2. **SG-LLIE / Structure-Guided Transformer Design** (2025),
3. **SNR-Aware Low-Light Enhancement** (CVPR 2022),
4. **LLFormer** (AAAI 2023),
5. **Zero-DCE++** (zero-reference lightweight baseline).

Do not add a sixth method in this cycle. If one of the five lacks a verifiable official implementation/checkpoint for the requested use, record it as unsupported rather than substituting an unofficial fork after inspection.

## Fixed inputs and settings

1. **Do not decode or run the official 100-pair LOL-v2 Real test set in this task.** No model inference, no test metrics, and no method tuning are authorized here.
2. Keep our accepted benchmark conventions fixed for the future comparison: native-resolution RGB images in `[0,1]`; full-frame RGB PSNR; Gaussian-window RGB-SSIM with the accepted 11×11, `sigma=1.5`, `K1=0.01`, `K2=0.03` convention; no crop, Y conversion, per-image normalization, or reference-based brightness/mean matching unless a separate paper table explicitly labels such a protocol as non-comparable.
3. For each of the five baselines, bind the exact official repository/project URL and exact inspected commit/tag; paper venue/year; implementation license; supported LOL-v2 Real training/test recipe; existence and provenance of official pretrained weights; input resizing/cropping/padding behavior; output range/color convention; reported metric implementation; and any post-processing.
4. Explicitly determine whether the published/repository inference or scoring path ever reads normal-light target pixels, target mean/statistics, PSNR/SSIM, or any reference-derived quantity to alter the enhanced output. **Any such target-assisted setting is inadmissible for our main comparison.** Retinexformer in particular must be checked for the repository-documented setting that uses the ground-truth mean; bind a target-free setting separately if one exists rather than quoting the target-assisted number as directly comparable.
5. Fairness-classify each baseline for our two-stage benchmark design:
   - `FINAL_TEST_READY`: official checkpoint/recipe may legitimately be evaluated on the untouched official LOL-v2 Real test because its training uses only the official training set and inference is target-free;
   - `VALIDATION_RETRAIN_REQUIRED`: an official checkpoint was trained on all 689 official training pairs and therefore would leak our frozen 100-pair validation subset; it may still be final-test-ready later, but cannot be used as a fair validation comparator unless retrained on the 589 non-validation pool;
   - `REJECT_TARGET_ASSISTED`: the candidate comparison recipe needs target/reference statistics at inference/scoring in a way that changes output or selection;
   - `UNSUPPORTED`: official code/checkpoint/provenance is not sufficiently verifiable.
6. Record published LOL-v2 Real PSNR/SSIM only as **context**, alongside the exact evaluation convention used. Do not mix incompatible reported numbers into an experimental table and do not treat literature numbers as reproduced results.
7. Freeze the future main-table rule: our final Ours configuration must be selected using only train/validation information before official-test decoding; once Ours is frozen, the official test is evaluated once under the same target-free metric pipeline used for admissible baselines. Normal-light test references are scoring-only after all outputs are frozen.
8. No changes to T022-C/T023-A code, weights, gate, action bounds, energy, learning rate, steps, or selector in this task.

## Acceptance / stop criteria

T024-A is **complete** iff all five named baseline families receive a provenance-backed fairness classification and the future main-comparison protocol is frozen in machine-readable form.

Call the protocol **benchmark-ready** only if at least three of the five have verifiable official code plus a target-free final-test recipe, including at least one 2025-or-newer method and at least one established 2022–2023 strong supervised method. Otherwise end `baseline coverage insufficient` with the exact missing pieces; do not compensate by adding unofficial forks or extra methods in this cycle.

If a repository documents multiple metric/test modes, preserve all relevant modes but designate exactly one target-free main-comparison mode. Do not choose a mode because its number is larger.

## Explicit non-goals

No official-test decoding; no baseline inference; no baseline training/retraining; no new Ours tuning; no 589-pair T023 scaling; no gamma/EV/LR/step sweep; no LPIPS computation yet; no downstream detector; no adaptive-geometry work. This task is protocol/provenance convergence only. Never authorize target-assisted output normalization for the main comparison.

## Expected evidence

Produce one compact T024-A package containing `research_log/T024A_baseline_protocol.md` plus a machine-readable JSON/CSV manifest. For each of the five baselines include exact official source/commit/tag, paper identity, checkpoint/training-data provenance, preprocessing/inference/post-processing, target-access audit, metric convention, license, fairness classification, and published LOL-v2 contextual numbers with a comparability warning. Include a final frozen main-table protocol and a prioritized execution order for later cycles. Append a concise completion report to `coordination/CODEX_TO_CHATGPT.md` ending exactly `benchmark-ready` or `baseline coverage insufficient`.

Never modify `coordination/PROJECT_STATE.md`. Stop after T024-A; the next research-lead review will decide whether the following one-hour cycle should reproduce one admissible baseline, run an action-space ceiling diagnostic, or resume method tuning.