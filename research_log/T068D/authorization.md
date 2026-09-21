# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task. Prior specifications/evidence remain in Git history and `research_log/`.

---

# Research-lead review — T068-C accepted as `TAIL_INEFFICIENCY_SIGNAL_ABSENT`

I reviewed main report commit `cc6ba251a8404111ca131b24a912d55662cd3d15`, PR #158, scientific source `dc9f766c16ad25971bb1a36a52eedc0a035733c2`, evidence/head `97216f4ada048f18b4b6b6301ae7eb2c5e092e3c`, `coordination/CODEX_TO_CHATGPT.md`, and the task-owned `research_log/T068C/**` implementation/evidence against the T068-C contract and `PROJECT_STATE.md`.

T068-C is scientifically accepted as **SIGNAL_ABSENT**. The exact three-transition endpoint statistic does not isolate the remaining unsafe T067-C endpoint. Development-only `T99=0.6324473434957039` is frozen from the 100 target-free development scores. On exposed transfer, the only unsafe selected endpoint (index 86, `k_FS=9`, `k_lambda=21`, margin vs T026 `-6.9954829928 dB`) has `R=0.4650582340`, rank `15/100`, and is **below** T99; meanwhile 2/99 safe endpoints exceed T99. Thus unsafe-above-threshold is `0/1`, so no guard is justified from this statistic.

The information boundary is valid. Development T99 uses no reference quality. The complete 100-row transfer score/component/state/output table was frozen with `reference_reads=0` before the first transfer quality read. Only after that freeze were the already-exposed T067-C/T066-B endpoint safety labels joined for diagnosis. The independent verifier rerendered both cohorts, recomputed RMS motion and scores with an independent CPU float64 implementation, reproduced T99/flags/diagnosis to numerical precision, and reports `optimizer_runs=0`, `model_fits=0`. No fresh cohort, official LOL-v2 Real test, LSRW, UHD-LL, test label, or clean target entered adaptation or checkpoint selection.

Scientific implication: T068-B already showed that **cumulative** scalar-objective-vs-image-motion geometry is an aggressive early-stop signal; T068-C now shows that this exact **tail-local** scalar-objective-vs-image-motion ratio also fails to identify the rare transferred late tail. Close this exact motion/total-objective signal family for now. Do not tune the window, percentile, ratio, tie rule, or turn T99 into a selector on the exposed cohort.

PR #158 is a stacked evidence PR. Review/retain the task-owned T068-C source and evidence rather than treating its historical aggregate diff as a merge recommendation.

---

# OPEN one-hour task — T068-D: frozen low-only objective-component Pareto-regret diagnosis

## Single hypothesis / engineering objective

Diagnose whether the residual unsafe `lambda=0.875` endpoint is instead characterized by **component conflict inside the fixed low-only objective**: the weighted scalar objective may continue to look acceptable while one or more constituent losses (`L_spa`, `10 L_exp`, `5 L_col`) have already moved away from their best values reached earlier in `[k_FS,k_lambda]`.

This is one diagnostic only. Do **not** build or evaluate a new stopping/rollback selector in this cycle.

## Fixed inputs/settings

Reuse exact accepted artifacts; do not rerun optimization or refit any model:

- T066-A model/features/normalization and probability threshold `0.5`;
- `rho=0.9857470621423519`, exact `k_FS`, and T063-C progress conventions;
- frozen T067-B/T067-C `lambda=0.875` endpoint `k_lambda` for each image;
- exact frozen `k=0..27` states/outputs and the existing per-step low-only component values;
- exact fixed objective decomposition and weights from T062: `[L_spa, 10 L_exp, 5 L_col]`;
- unchanged renderer, action space, objective weights, Adam settings and trajectories;
- the original 100-image development cohort plus the already-exposed T063-D/T067-C transfer cohort only. Fresh/final sets remain sealed.

For each image and each step `k in [k_FS,k_lambda]`, define the weighted component vector in float64:

`Z_k = [L_spa(k), 10*L_exp(k), 5*L_col(k)]`.

For each component `c` over this frozen interval define:

- `zmin_c = min_k Z[k,c]`;
- `zmax_c = max_k Z[k,c]`;
- endpoint regret `a_c = Z[k_lambda,c] - zmin_c`;
- available excursion `e_c = zmax_c - zmin_c`.

Use exactly one scalar score:

`R_comp = sum_c a_c / max(sum_c e_c, 1e-12)`.

If `k_FS == k_lambda` or `sum_c e_c <= 1e-12`, set `R_comp=0` and record the reason. Do not add any other normalization, component weighting, clipping, smoothing or fallback.

First compute 100 **development** scores using only these target-free quantities. Freeze one development-only threshold exactly as nearest-rank 99th percentile: sort the 100 finite `R_comp` values ascending and set `T99_comp = sorted_R[98]` (0-based index 98). Do not use development PSNR/SSIM, safety labels or clean targets to form this threshold.

Then compute the 100 **exposed-transfer** endpoint scores with the same formula and frozen `T99_comp`. Before reading/joining any transfer reference-quality information in this task, freeze/hash the complete 100-row transfer table containing input identity, `k_FS`, `k_lambda`, all three endpoint weighted components, `zmin/zmax`, `a/e`, `R_comp`, strict `R_comp>T99_comp`, state/output/rule/input hashes, and `reference_reads=0`.

Only after that freeze may you join the already-exposed T067-C/T067-D endpoint safety label `unsafe := (PSNR(k_lambda)-PSNR_T026 < -5.614 dB)` for diagnosis. Do not recompute or change any checkpoint choice.

## Acceptance / stop criteria

Return exactly one diagnosis:

- `COMPONENT_REGRET_SIGNAL_PRESENT` iff **all** unsafe transfer selected endpoints have `R_comp>T99_comp` **and** at most `5` safe transfer selected endpoints have `R_comp>T99_comp`;
- `COMPONENT_REGRET_SIGNAL_ABSENT` otherwise;
- `BLOCKED` on any source/cohort/state/output/component/hash mismatch, if the transfer score table is frozen after transfer reference-quality access, if T99 is not computed exactly from the target-free development scores above, or if the independent verifier disagrees.

If the exact frozen T067-C labels contain zero unsafe selected endpoints, return `BLOCKED`. Stop immediately after the diagnosis. A positive diagnosis is **not** authorization to deploy a guard in this same cycle.

## Explicit non-goals

No component subset/weight sweep, no alternate aggregation (`max`, product, learned combination, etc.), no percentile/threshold/window sweep, no learned classifier, OOD model, k-NN, smoothing, patience, rollback, selector, lambda/rho/probability-threshold change, objective-weight change, optimizer rerun, model refit, trajectory extension, or baseline/gate change. Do not evaluate PSNR/SSIM for any hypothetical component-regret-guarded output.

Do not open any fresh cohort, official LOL-v2 Real test, LSRW, UHD-LL, or final benchmark set. Test-time adaptation/selection must consume **no test labels, clean/normal-light targets, PSNR/SSIM, oracle safe ranges/boundaries, degradation annotations, semantic IDs, or per-image baseline outcomes**.

## Expected evidence

Commit the exact source SHA and binding manifests; focused tests for weighted component extraction, interval minima/maxima, zero-excursion/singleton handling, exact `R_comp`, strict thresholding and nearest-rank `T99_comp`; a target-free 100-row development table and frozen T99; a pre-reference 100-row transfer score freeze with hash/timestamp and `reference_reads=0`; first transfer-reference-quality-read timestamp; joined endpoint safety table; counts/ranks for unsafe and safe endpoints above T99; explicit component/regret row for every unsafe endpoint; independent verifier output that recomputes the three low-only components directly from degraded image plus frozen rendered states (not by trusting the primary score table), checks hashes/threshold/diagnosis, and confirms `optimizer_runs=0`, `model_fits=0`; run receipt; and one concise completion report appended to `coordination/CODEX_TO_CHATGPT.md`.

Never modify `coordination/PROJECT_STATE.md`; stop after reporting.
