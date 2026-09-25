# TTIE Long-Horizon Execution Plan

**Authority:** user-approved autonomous execution plan, 2026-09-25.  
**Goal:** finish the paper-grade cross-domain comparative main table first, then the preregistered ablation and analytical experiment package, without waiting for a new one-hour instruction after every successful subtask.

## 0. Operating mode

Codex may now continue autonomously from one completed subtask to the next along this document. The hourly ChatGPT loop becomes a **research-lead audit / intervention loop**, not a permission gate after every normal success.

Priority order is immutable unless the user or research lead explicitly overrides it:

1. Repair and seal the UHD-LL expanded preregistration.
2. Finish every UHD-LL main-table output row and freeze/hash it.
3. Open UHD-LL references only after the required output gate passes; compute only the preregistered metrics/statistics.
4. Repeat the same source→target protocol on LSRW.
5. Preregister and run Target-3, then freeze the combined three-target main table.
6. Only then execute the ablation program.
7. Then execute the analytical/diagnostic program and paper-ready figures/tables.

Do **not** reopen in-domain rescue, redesign Final Ours, tune from held-out outcomes, poll the retired A6000 host, or spend time on unrelated side analyses.

If one independent method is blocked, record a fail-closed BLOCKED state and continue other independent main-table rows rather than idling. However, target references must remain sealed until every required Tier-1 row is either successfully frozen or has a prospectively justified terminal reproducibility classification accepted by the research lead.

The newest explicit instruction in `coordination/CHATGPT_TO_CODEX.md` overrides this plan only where they conflict.

---

# 1. Absolute information boundary

For every target dataset:

- Test-time adaptation, checkpoint/source selection, state selection, stopping, retry, sample inclusion, execution rescue and hyperparameter choice may use **target-low data only** plus source/frozen assets.
- They may never use target clean/normal-light images, test labels, PSNR/SSIM/LPIPS, baseline outcomes, oracle quality, reference-derived metadata, or qualitative comparison against references.
- Before the output-freeze gate, target references must not be enumerated, resolved, opened, decoded, cached or passed through dataset objects.
- Every target-time runner must have an auditable `reference_reads=0` receipt.
- Output inspection itself must not be used to tune settings. A smoke may verify only execution properties: geometry, dtype, finiteness, hashes, runtime/VRAM, parameter/state integrity.
- Once a method's complete outputs are frozen for a target, do not rerun that method because later metrics look poor.
- Credentials/secrets never enter GitHub, committed logs, PR text or coordination files.

A target reference may be opened only after the target-specific method/output registry passes the frozen-output gate.

---

# 2. Fixed method roles and reporting schema

## Tier-1 — required

1. RetinexFormer
2. SNR-Aware
3. PromptIR
4. PromptIR + DCTTA
5. MR. Illuminate
6. QuadPrior
7. Ours-Step0 / Ours w/o TTT
8. Ours-TTT

## Additional

- ZERO-IG — strongly preferred additional row.
- GM-MoE — secondary generalization row; should be run if official reproducibility is practical but must not delay Tier-1.

For every row record:

- exact method ID/display name;
- paper/repository identity and code commit if available;
- exact source-training dataset / official checkpoint setting;
- checkpoint/config/source hashes;
- paradigm: fixed / domain-level TTA / per-image zero-shot / per-image episodic TTT;
- target-low access scope;
- adaptation granularity and reset/carry policy;
- target GT used during adaptation: **No** for every valid row;
- preprocessing and output geometry policy;
- mean PSNR, median PSNR, mean RGB-SSIM after reference gate;
- paired PSNR delta + win fraction where preregistered;
- adapted-minus-base gain for PromptIR+DCTTA vs PromptIR and Ours-TTT vs Ours-Step0;
- adaptation time, inference time, peak VRAM;
- test-time updated parameter/state size;
- execution/reproducibility notes.

Do not imply matched-source training. The table must visibly disclose heterogeneous official source settings.

---

# 3. Phase A — finish T073-A-R1 and seal UHD-LL preregistration

This is the immediate starting gate.

Repair the current T073-A branch before any new target model run:

1. Exact Tier-1 is the eight methods listed above.
2. ZERO-IG = preferred additional; GM-MoE = secondary.
3. Preserve T072-L metric provenance, exact canonical 150-image UHD-LL cohort, complete-case policy, shared paired-bootstrap seed `20260922`, 10,000 resamples, sign conventions and fail-closed rules.
4. Preserve accepted T072-AZ-B RetinexFormer/SNR-Aware manifest SHA256:
   `73c8d304c8bb88c4612a13598b5d4dd76378e0c3107bd97011941037a4427001`.
5. Add `LOW_ONLY_DCTTA_REQUIRED`: DCTTA may not enumerate/open/cache target GT.
6. Extend verifier/mutation tests for mis-tiering, base-checkpoint mismatch, unresolved source, GT-read permission, changed metric provenance/seed/count/cohort.

### Resolve PromptIR/DCTTA source binding prospectively

Use the official DCTTA low-light-relevant setting **before any UHD-LL PromptIR/DCTTA model run**:

- Official DCTTA `model.ckpt` = three-task PromptIR (rain/haze/noise).
- Official `epoch=80.ckpt` = five-task PromptIR.
- The DCTTA CVPR 2026 paper states that the five-task setting extends the three-task setting with deblurring and **low-light enhancement**, using LOL for low-light training and LOLv2-real for low-light cross-distribution evaluation.

Therefore, for this low-light source→target comparison, bind **both PromptIR and PromptIR+DCTTA to the same official five-task `epoch=80.ckpt`**, with the exact artifact hash recorded. This choice is task/provenance-driven and is made before UHD-LL outcomes.

Do not compare PromptIR 3-task against DCTTA 5-task. Their base must be identical.

Seal T073-A-R1 only after all verifier/mutation tests pass with:
`reference_reads=0, metrics=0, model_runs=0`.

After seal, continue immediately to Phase B.

---

# 4. Phase B — UHD-LL complete main-table outputs

Canonical target: the already sealed 150-image UHD-LL low-only cohort, native 3840×2160 source images.

## B1. RetinexFormer — DONE/FROZEN

Use the accepted T072-AZ-B output manifest only. Do not rerun.

## B2. SNR-Aware — DONE/FROZEN

Use accepted T072-AZ-B outputs with the prospectively validated 512-query-row execution-equivalent memory schedule. Do not rerun.

## B3. PromptIR static + PromptIR+DCTTA

### B3a. Low-only binding audit

Before target run:

- pin official DCTTA repository/code;
- pin official five-task `epoch=80.ckpt`;
- hash checkpoint/code/config;
- prove PromptIR static and DCTTA share exactly the same base weights;
- inspect official DCTTA loader/adaptation path and identify all target-GT touches;
- implement a task-owned **low-only wrapper** that supplies exactly the degraded-image tensors/metadata consumed by adaptation while never resolving a GT path;
- test wrapper equivalence on synthetic/source-side paired examples: degraded-input tensor and adaptation-relevant computation must match the original path when GT is withheld from the actual computation;
- run mutation tests showing any GT path/open call fails closed.

DCTTA paper-defined adaptation components should remain unchanged: test-time redegradation, degradation-consistent/self-supervised restoration adaptation, and important-parameter selection. Do not simplify DCTTA merely for convenience.

### B3b. Smoke and full run

- one execution-only smoke for PromptIR static and one for DCTTA;
- smoke checks only geometry/dtype/finiteness/runtime/VRAM/hash/state integrity;
- no metric or reference read;
- if valid, execute complete 150 PromptIR outputs and complete 150 DCTTA outputs;
- record DCTTA adaptation granularity/reset policy exactly as the official method uses it. If domain-level/cumulative rather than episodic, preserve it and disclose it; do not force it into our episodic protocol.
- freeze/hash all outputs and independent verification.

If native 4K causes an OOM, first inspect official high-resolution inference behavior. Any memory-only scheduling rescue must be prospectively sealed and proven computation-equivalent before one target retry, following the SNR-Aware precedent. No result-driven precision/resize/tile rescue.

## B4. Ours-Step0 + Ours-TTT

Use immutable T070-A scientific state:
- CommonRegion2/CommonBox 12-D EV/gamma/gain renderer from identity;
- 27 float32 Adam updates, lr=0.03;
- `L_spa + 10 L_exp + 5 L_col`;
- frozen T066-A 19-D safety model, threshold 0.5;
- `rho=0.9857470621423519`;
- T067-B `lambda=0.875` with exact historical endpoint/tie semantics.

### Ours-Step0
Run the exact frozen T070-A forward/output path at step 0 with **zero optimizer updates**. Do not redefine “w/o TTT” to make it stronger. If the frozen implementation makes step-0 identical to the low input, document that fact transparently.

### Ours-TTT
Use exact T070-A; no retuning. Verify existing UHD-LL outputs if already complete and provenance-valid; otherwise run complete 150. Freeze/hashes independently.

Required paired analysis later: `Ours-TTT − Ours-Step0`.

## B5. MR. Illuminate

Treat as a zero-shot/generalization competitor. Its published method is Modulate-Refine with pretrained diffusion prior and requires no per-target optimization.

Execution:
- use official code/weights/default inference path;
- pin version/hashes;
- no target training/tuning;
- preserve official preprocessing;
- perform one execution smoke then complete 150;
- output must be deterministically mapped to the canonical evaluation geometry under the official protocol; document any official resizing;
- freeze/hash and verify.

## B6. QuadPrior

Treat source setting explicitly: official QuadPrior is trained using COCO images and is zero-reference with respect to paired low/clean supervision.

Execution:
- use official released code/checkpoint and default inference;
- pin source/checkpoint hash and COCO-source provenance;
- no UHD-LL tuning;
- one smoke, then 150 full;
- record any official resolution/preprocessing;
- freeze/hash/verify.

Do not substitute QuadPrior++ unless the research lead explicitly changes the row; the main row is QuadPrior.

## B7. ZERO-IG — preferred additional

Official repository notes that provided weights are not the exact paper-result model and recommends single-image training for better results. Therefore reproduce the **officially documented per-image zero-shot procedure** rather than pretending the shipped weights equal paper results.

Before execution:
- freeze exact official training/inference defaults from code/README;
- adaptation may use only that image's target-low pixels;
- reset independently per image;
- no GT/reference;
- no per-image stopping chosen from metrics/visual reference.

Run one low-only smoke and then all 150 if feasible. Record per-image adaptation time and updated parameter count. If official code cannot reproduce a paper-faithful protocol without undocumented choices, classify transparently as `BLOCKED_REPRODUCIBILITY`; do not invent a tuned variant.

ZERO-IG failure must not block Tier-1 reference gate once its inclusion status is prospectively locked.

## B8. GM-MoE — secondary

Use official released checkpoint/config if available. Treat as fixed generalization-oriented LLIE, not TTA. Pin training/checkpoint provenance and run one smoke → complete 150 → freeze/hash. If official checkpoint is unavailable or ambiguous, record `BLOCKED_REPRODUCIBILITY` and continue; this row must not delay Tier-1.

---

# 5. Phase C — UHD-LL reference gate and final table

Only when all eight Tier-1 rows have complete frozen outputs and ZERO-IG/GM-MoE inclusion statuses are locked:

1. run independent output-registry verifier;
2. generate one immutable gate receipt listing method IDs, source hashes, output manifest hashes, exact 150 names, geometry and inclusion status;
3. only then unlock/reference the UHD-LL clean target payload;
4. compute the already-preregistered RGB PSNR/RGB-SSIM implementation only;
5. no model reruns after seeing metrics.

Report:
- mean PSNR;
- median PSNR;
- mean RGB-SSIM;
- Ours-TTT − each comparator paired PSNR mean/median where specified;
- win fraction;
- shared 10,000-resample paired-bootstrap CIs using seed 20260922;
- PromptIR+DCTTA − PromptIR gain;
- Ours-TTT − Ours-Step0 gain;
- runtime/VRAM/state-size columns from frozen receipts.

Produce both machine-readable results and a paper-ready table. Do not change method inclusion because of poor scores.

---

# 6. Phase D — LSRW source→target replication

Before reading LSRW references:

1. identify the canonical complete paired LSRW test split from the authorized archive;
2. seal sample IDs, low hashes, reference hashes in a **reference-opaque** manifest (the execution process must not open reference payloads);
3. preregister the same method registry, source settings, metric implementation and statistics;
4. no source checkpoint changes because of UHD-LL outcomes.

Run the same methods in the same role:
RetinexFormer, SNR-Aware, PromptIR(5-task), PromptIR+DCTTA(5-task), MR. Illuminate, QuadPrior, Ours-Step0, Ours-TTT, plus ZERO-IG and GM-MoE according to their already-locked inclusion status.

Each method:
smoke → full low-only outputs → freeze/hash → independent verify.

Reuse previously validated execution-equivalent memory schedules where geometry requires them; do not retune schedules from LSRW quality.

Only after the complete LSRW output gate passes, open references and compute the same preregistered metrics/statistics.

---

# 7. Phase E — preregistered Target-3

Do not select Target-3 from observed performance.

Target-3 selection criteria, in order:
1. paired real low-light/normal-light benchmark;
2. distinct capture/degradation distribution from LOL source, UHD-LL and LSRW;
3. canonical complete test split and stable public protocol;
4. sufficient reproducibility for all Tier-1 methods.

Preferred candidate: **SDSD-Indoor official test split** if the canonical paired split is available and compatible. If unavailable, perform a provenance-only feasibility comparison against another paired real LLIE dataset (e.g. SID only if its raw/sRGB protocol is truly compatible), and lock the choice **before any method outcome/reference metric is inspected**.

Write a Target-3 preregistration specifying cohort, exact method/source registry, preprocessing policy, metric/statistic plan and information boundary. Then run the same output-freeze → reference-gate → metrics pipeline.

After UHD-LL + LSRW + Target-3 are complete, create the final three-target comparative main table and a compact cross-target average/rank-free summary. Do not turn heterogeneous datasets into an unjustified single scalar “winner score”.

---

# 8. Phase F — ablation program

Ablations must not be used to retune Final Ours. Final Ours remains T070-A.

Use an existing **development/exposed transfer cohort** that is already designated mechanism/diagnostic evidence and excluded from the final main-table claims. If a new ablation cohort is needed, preregister it from non-final data before running ablations. Do not use the three held-out final target metrics to choose ablation settings.

## F1. Core contribution ablation table

Run the following fixed variants under identical low-only conditions:

### A0 — Ours-Step0
No TTT update.

### A1 — Global-only ISP
Replace spatially varying CommonRegion2/CommonBox adaptation with the closest scientifically valid **global EV/gamma/gain** parameterization while preserving optimizer, loss weights, update count and selection logic. This tests whether spatial variation is necessary.

### A2 — Spatial TTT, fixed-final selection
Full spatial renderer + full low-only objective, but always use step 27 instead of the safety/utility selector. This isolates target-free state selection.

### A3 — Spatial TTT, selector without safety–utility interpolation
Use the exact historical T067-B endpoint corresponding to “no interpolation” according to code semantics. Do not guess which lambda endpoint means safety-only/utility-only; derive and document it from the frozen implementation.

### A4 — Full Ours
Exact T070-A.

Primary ablation report: mean/median PSNR, SSIM, failure/win fraction against A0 on the **development ablation cohort**, plus runtime.

## F2. Objective-component ablation

With all other settings frozen:
- remove `L_spa`;
- remove `L_exp`;
- remove `L_col`;
- full `L_spa + 10 L_exp + 5 L_col`.

When removing a term, do not renormalize the remaining weights. This isolates contribution of each self-supervised signal.

## F3. Selection hyperparameter sensitivity — development only

Use fixed values chosen prospectively:
- lambda: include the historical endpoints and `0.5`, plus frozen `0.875`;
- safety threshold: `0.3, 0.5, 0.7`;
- maximum update budget: `9, 18, 27`.

These are **sensitivity analyses only**. Do not choose a new Final Ours from them.

## F4. Episodic reset ablation

Compare:
- frozen episodic per-image reset (Final Ours);
- a clearly labeled carry-over state variant across images in a fixed lexicographic order.

Use only the development cohort. Purpose: show whether episodic reset prevents order/domain drift. Do not use this to alter Final Ours.

---

# 9. Phase G — analytical experiments

## G1. TTT trajectory dynamics

For the development cohort, store every step 0…27:
- `L_spa, L_exp, L_col, L_total`;
- parameter/state vector;
- safety probability / first-safe event;
- selected step;
- development-only PSNR/SSIM for analysis.

Produce:
- mean quality-vs-step curve;
- mean target-free objective-vs-step curve;
- selected-step histogram;
- first-safe vs selected vs final-step quality comparison.

Goal: explain why “more adaptation” is not always better and why target-free selection matters.

## G2. Spatial heterogeneity analysis

From low-only inputs compute a preregistered heterogeneity index, e.g. variance of local log-luminance/exposure across the same spatial regions used by the renderer. Do not tune its definition from quality outcomes.

For each image compute:
`gain_spatial = quality(Global-only ISP) − quality(Step0)`
and
`gain_full = quality(Full spatial TTT) − quality(Global-only ISP)`.

Analyze gain versus heterogeneity using:
- Spearman correlation;
- predeclared low/mid/high heterogeneity tertiles;
- mean/median gain with bootstrap CI.

Also visualize the learned EV/gamma/gain regional states for preselected examples.

## G3. Safety/selection reliability

On development data:
- first-safe step distribution;
- fraction where step 27 is worse than selected step;
- regret of selected step relative to per-image oracle best step **for analysis only**;
- oracle is never allowed into runtime selection.

Report selector regret distribution and catastrophic-tail avoidance rate. Clearly label oracle quantities post-hoc diagnostic only.

## G4. Cross-target TTT benefit

Using frozen final main-table outputs after each reference gate:
- paired gain `Ours-TTT − Ours-Step0` per image;
- distribution/CDF/box or violin for UHD-LL, LSRW, Target-3;
- mean/median, win fraction and frozen bootstrap CI;
- no retuning from these plots.

This is the key evidence that TTT helps under unseen degradation.

## G5. TTA competitor comparison

Compare **gain from adaptation**, not only absolute quality:
- DCTTA gain = PromptIR+DCTTA − PromptIR;
- Ours gain = Ours-TTT − Ours-Step0.

For each target show:
- paired gain distribution;
- adaptation time;
- target-low access scope;
- domain-level vs per-image granularity;
- updated parameter/state size.

Do not claim source-matched fairness if base architectures/training differ.

## G6. Efficiency / deployment analysis

Create a table:
- adaptation time/image or domain;
- inference time;
- peak VRAM;
- updated parameter count/state bytes;
- whether source data is required at test time;
- whether target GT is used;
- whether state resets per image.

Use measured receipts, not estimates where direct measurement is possible.

## G7. Low-only characteristic stratification

Before outcome analysis, fix low-only characteristics:
- mean luminance;
- luminance contrast;
- saturation fraction;
- local illumination heterogeneity;
- simple noise proxy.

Bin each characteristic into predeclared tertiles based only on target-low values. After references are legally open, report Ours-Step0→TTT gains by bin. This links gains to degradation severity without using GT to define the bins.

## G8. Qualitative figure

Main qualitative samples must be selected **without reference metrics**:
- either fixed lexicographic IDs, or
- fixed low-only tertile sampling from the above characteristics.

Show low input, key baselines, Ours-Step0 and Ours-TTT, with identical crops. Do not cherry-pick by PSNR.

A separate clearly labeled failure-analysis panel may show worst-k Ours-TTT cases after metrics are available; it must not be used as the main qualitative comparison.

---

# 10. Evidence, Git and communication requirements

For every subtask/phase:

- create a collision-free `research_log/<TASK>/`;
- include human-readable report;
- machine-readable config/manifest;
- hashes of code/checkpoints/cohort/outputs;
- GPU/environment receipt when relevant;
- verifier and verifier receipt;
- `reference_reads`, `metrics`, and `model_runs` counters/classification;
- append concise milestone result to `coordination/CODEX_TO_CHATGPT.md`;
- update `coordination/PROJECT_STATE.md` only for genuine accepted scientific-state changes.

Commit after meaningful milestones; do not accumulate an unauditable mega-commit.

## Autonomous continuation rule

After a subtask passes its predeclared acceptance criteria, immediately proceed to the next item in this plan without waiting for another research-lead message.

Stop and request intervention only when:
- a Tier-1 method has an unresolved scientific/protocol ambiguity;
- information-boundary safety cannot be proven;
- a required dataset/checkpoint is unavailable;
- an execution rescue would change scientific semantics;
- contradictory evidence invalidates the preregistration;
- credentials/hardware authorization is missing.

Ordinary dependency installation, code porting, deterministic wrappers, hashing, smoke tests, full runs, verification, and report generation do **not** require a new hourly authorization.

## Research-lead hourly loop

ChatGPT will inspect progress every hour. Unless it posts an explicit correction/stop, Codex should keep executing this long-horizon plan.

**End state:** three-target source→target main comparison complete; ablation table complete; analytical experiments/figures complete; all claims traceable to frozen evidence; no held-out leakage.
