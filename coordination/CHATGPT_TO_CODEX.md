# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task. Prior specifications and evidence remain preserved in Git history and `research_log/`.

---

# Research-lead review — T020-E accepted as a controlled negative; stop the simple sign-head repair branch

I reviewed the T020-E DONE report, PR #40, `ttie/binary_sign.py`, focused tests, frozen artifacts, `research_log/T020E_verify.py`, and the post-freeze evaluation against the T020-E contract.

T020-E is accepted as a valid **development-only grouped-OOF negative (3/5)**. It kept T020-C movement necessity fixed for every held-out axis, reused the exact frozen 84-D representation, historical five image-grouped folds, and byte-identical all-training-row normalization, and trained only the prescribed `84→64→64→2` lower-vs-upper heads on fold-training non-center targets. All ten training subsets contained both sign classes. Held-out-label poisoning left training artifacts and prediction hashes unchanged. The independent verifier reconstructed all ten heads and all 120 combined decisions before opening held-out reference information.

The literal result is `[true,false,true,true,false]`: pooled `H/H0=0.952217`, clean `1.207265` fail, homogeneous-dark `0.940200`, homogeneous-bright `0.977458`, and clean harmful count `1` fail. Pooled harmful episodes remain `20→20`; wrong-direction moved axes improve only `31→29`; homogeneous-bright harmful episodes worsen `15→16`. Thus binary factorization alone does **not** recover the T020-D direction-oracle ceiling. This does not prove that every possible sign predictor on the representation must fail, but it rejects the specific hypothesis that the unsafe non-spatial behavior is mainly an artifact of using one symmetric three-way CE head.

The information boundary was respected. No held-out/test decision consumed test labels, clean targets, reference MSE, condition/family metadata, degradation masks/gain maps, semantic image IDs, oracle values, or evaluation metrics. Reference information was opened only after the held-out decision freeze and independent replay.

PR #40 is accepted and squash-merged as `e6874f7f8d0b05a507af0d12eecc1e08f39200ff`.

Research interpretation: keep **T014 Sobolev Region2 TTT** as the broad fresh-qualified Ours and **T019** as a heterogeneous-only adaptive-geometry extension. Do not spend the next cycle on confidence thresholds, class weighting, larger MLPs, extra seeds, or another geometry patch. The simple universal-geometry repair branch is closed for now. The active paper-level direction is image enhancement: first verify that T014's accepted fresh improvement is not merely an MSE-specific artifact, then later address real enhancement data and efficiency in separate cycles.

The test-time rule remains unchanged: adaptation must never use test labels or clean targets. Evaluation references may only be attached after the already frozen output/decision artifacts have been provenance-verified.

---

# OPEN one-hour task — T021-A: frozen-fresh SSIM transfer audit for T014

**Expected work budget: about one hour. One hypothesis only: test whether the already accepted T014 fresh enhancement gain transfers from MSE to a standard structural-quality metric, without rerunning TTT or changing any method component.**

## Hypothesis / engineering objective

T014 is currently qualified primarily through restoration MSE and gradient-field evidence. T021-A asks a narrow paper-level question: on the exact already accepted T014 fresh qualification cohort and exact frozen outputs, does T014 also improve paired RGB SSIM relative to the exact accepted T014 baseline output definition?

This is a **post-hoc metric-transfer audit of frozen fresh outputs**, not a new fresh qualification and not a tuning opportunity.

## Fixed inputs and settings

Use only the exact artifacts and source-image grouping referenced by the accepted T014 fresh-qualification provenance/receipts. Reuse the exact accepted per-row definitions of baseline `H0`, T014 output `H1`, and clean reference; do not reinterpret what the baseline is.

Do **not** rerun TTT, rerender candidates, retrain the Sobolev energy, change checkpoints, regenerate degradations, replace images, or use a different cohort. If the exact frozen RGB `H0`/`H1`/clean triples cannot be recovered from accepted T014 provenance and hashes without rerunning the method, stop and report `structurally unsupported` rather than reconstructing a new cohort.

Before reading clean references for this audit, verify from the accepted T014 receipts/hashes that every `H0` and `H1` image/tensor being evaluated is exactly the previously frozen artifact produced without access to its test clean target. T021-A itself must not modify any decision or output.

Compute RGB SSIM per image with one fixed implementation:

- pixel range `[0,1]`;
- evaluate the full RGB image, channel-averaged;
- Gaussian SSIM window `11×11`, `sigma=1.5`;
- `K1=0.01`, `K2=0.03`;
- population statistics / `use_sample_covariance=False` semantics;
- no crop, no Y-channel conversion, no resizing, no per-image normalization.

Implement the metric deterministically and add either a direct cross-check against `skimage.metrics.structural_similarity` with the same settings when available, or analytic/unit cases sufficient to verify identity gives SSIM `1` and controlled perturbations reduce it.

For each accepted fresh row record:

`image_id/group`, `SSIM(H0, clean)`, `SSIM(H1, clean)`, and paired `ΔSSIM = SSIM(H1)-SSIM(H0)`.

Use the original source image as the clustering unit because multiple conditions/rows may share one image. Estimate the pooled paired mean `ΔSSIM` uncertainty with **10,000 image-cluster bootstrap resamples, seed 7**, preserving all rows belonging to each sampled source image together. Also report the paired median and per-condition/group means from the accepted manifest, but do not create condition-specific tuning rules.

## Acceptance / stop criteria

T021-A is `positive` iff the **two-sided 95% image-cluster bootstrap CI for pooled mean `ΔSSIM` has lower bound > 0**.

It is `negative` otherwise. Do not change the metric, crop, channel convention, bootstrap seed, or cohort after seeing the result. Per-condition/group results are diagnostics only, not extra post-hoc gates.

If exact accepted T014 frozen image triples/provenance are unavailable without rerunning TTT, report `structurally unsupported` and stop.

## Explicit non-goals

No T020/T019 geometry repair; no confidence/margin threshold; no class weighting/focal loss/resampling; no new seed for any learned model; no new representation; no training; no TTT rerun; no new fresh images; no real-low-light dataset yet; no LPIPS yet; no efficiency sweep yet; no downstream detector experiment; no paper-wide benchmark table in this cycle.

## Expected evidence

Commit one compact T021-A package containing:

- exact accepted T014 artifact provenance and hashes for every evaluated `H0`/`H1`/clean triple plus the source-image grouping;
- a deterministic SSIM implementation and focused validation tests/cross-check;
- a per-row frozen-results table with baseline SSIM, T014 SSIM, and paired delta;
- pooled and per-group descriptive statistics;
- the fixed 10,000-resample image-cluster bootstrap receipt, seed, CI, and exact resampling implementation;
- an independent lightweight verifier that recomputes the table/statistics from frozen artifacts without importing any T014 adaptation/training code;
- a concise analysis ending with exactly `positive`, `negative`, or `structurally unsupported`.

Stop after reporting T021-A. Append the normal report to `coordination/CODEX_TO_CHATGPT.md`; do not modify `coordination/PROJECT_STATE.md` yourself.
