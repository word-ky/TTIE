# Project State

## Research branch

**Spatially Varying Test-Time ISP / Image Enhancement**

This file is the live scientific state. Detailed task-by-task history remains in Git history and `research_log/`.

## Core scientific question

Can a vision system adapt a compact spatial image-processing state per test image, without test labels or clean/normal-light targets, so that unknown and spatially heterogeneous degradation is corrected by a reference-free test-time optimization field?

## Current deployable state

**T014 Sobolev Region2 TTT remains the broad fresh-qualified Ours-Core.** On LOL-v2 Real development data, the accepted fixed-validation deployable base remains **T026-A: `11.1208764 dB / 0.3737918 RGB-SSIM`**. T036-A common gain remains the strongest fresh-qualified deployable action expansion found so far, improving exact T026-A by `+0.9392571 dB / +0.0083560 RGB-SSIM` mean on its deterministic reference-unused development cohort, but with an unresolved unsafe tail (`29/100` PSNR regressions; worst `-5.614 dB`).

The accepted T036 common path starts independently from each raw low image with identity `raw ∈ R^{1×3×2×2}`, jointly updates all 12 EV/gamma/gain coordinates using Adam `lr=0.03` for 40 active steps plus CommonBox, and uses the original T014 scalar energy for both the full raw gradient and minimum-energy/earliest-tie checkpoint selection.

No T059/T060 action-transfer result is deployable. The T059/T060 rescue line is now closed after T060-D-R2. The official LOL-v2 Real test remains sealed.

## Renderer / action-family diagnosis

Reference-only renderer studies established that remaining capacity is strongly spatial:

- **T051-A spatial exposure field:** fixed 8×8 RGB-shared exposure adds `+0.6590052 dB` mean / `+0.4525454 dB` median PSNR over T050, with `100/100` PSNR wins.
- **T054-A local-detail field:** strongest later reference-only renderer extension, reaching `24.3454867 / 0.7577572` and adding `+1.2383356 dB` mean / `+1.2353360 dB` median PSNR plus `+0.1877170` mean SSIM over T052.
- T055-A/T055-V show that simply extending the same one-scale optimization budget adds only about `+0.0043 dB`; T056 second-scale detail and T057 chroma-detail failed their fixed materiality gates.

All of these are non-deployable `REFERENCE_ORACLE_ONLY` diagnostics. Their clean targets, reference gradients/states, PSNR/SSIM, and per-image oracle quantities are forbidden from test-time inference.

## T059 matched-detail optimization-field diagnosis — closed as a practical detail route

The T059 program established a stable asymmetry: **direction transfers much better than calibrated scalar value**, but the tested detail action family is not practically useful enough to continue tuning.

Accepted conclusions remain:

- calibrated unseen-image scalar prediction is unsupported despite strong in-sample capacity;
- simple 28-D locality, bank centering/context, early stopping, frozen-CLIP Euclidean locality, donor smoothing/oracles, dual-head disagreement, optimizer-scale rescue, and low-gradient-norm abstention do not solve the value/safety problem;
- the detail field itself is directionally real: T059-E source inner-held alignment is `0.868874` positive-dot / `0.491553` median cosine; T059-S/T/U show most source anchors have useful local descent direction; T059-V/W show the degraded-image Jacobian bridge and one-step action can be reconstructed online without reference-dependent caches;
- T059-X transfers to the already-open 100-image LOL-v2 Real development cohort from raw input, with `97/100` MSE improvements, but only `+0.001135428 dB` mean PSNR;
- T059-Y after exact T026-A remains numerically positive but materially negligible: `+0.002424174 dB` mean PSNR, `+0.001407602` RGB-SSIM, and `9/100` harmful cases.

Therefore the fixed one-step matched-detail integration line is closed.

## T060 action-family transfer diagnosis — closed

### T060-A — direct 8×8 spatial-exposure projection: negative

The exact T051-style exposure projection failed the preregistered coverage/shape audit: `65/80` nondegenerate versus required `72/80`; among the 65, positive-dot is `60/65 = 0.9231` but median cosine is `0.3524 < 0.40`. The direct full-field route is closed. The residue is a useful coarse descent sign, not reliable transferred high-dimensional spatial shape.

### T060-B — exact T036 common-gain subspace comparison: positive first-order result

On the exact same 60 nondegenerate common-gain source anchors, T059-E is substantially better aligned with true restoration direction than deployed T014:

- positive-dot: **T059-E `58/60 = 0.9667` vs T014 `53/60 = 0.8833`**;
- wrong-sign count: **`2` vs `7`**;
- cosine mean: **`0.7775` vs `0.6290`**;
- cosine median: **`0.9064` vs `0.8009`**, delta `+0.1055`;
- cosine p10: **`+0.4259` vs `-0.0607`**.

This remains accepted source-only first-order evidence that T059-E is a better direction candidate in the exact low-dimensional T036 common-gain subspace. It is not finite-step/deployable evidence by itself.

### T060-C-R1 — fixed gain-slice finite-step test: negative, but near-miss safety shift

T060-C-R1 executed the corrected intervention on the exact accepted T036 100-image development cohort. It is scientifically admissible: raw-low/identity initialization, all 12 optimizer coordinates, Adam `0.03 × 40`, CommonBox, T014 EV/gamma gradients, and T014 minimum-energy/earliest-tie selection are preserved; only the four gain-gradient entries are replaced by fresh target-free T059-E `J_gain^T q_E`. All 100 actions/states/outputs froze before any normal/reference or persisted per-image baseline outcome was opened. No clean target, label, reference gradient/Jacobian, PSNR/SSIM, harm label, or official-test information entered adaptation.

Frozen result versus exact persisted T026 on the same T036 cohort:

- mean paired PSNR: **`+0.920677 dB`** (`>= +0.80`, pass);
- median paired PSNR: `+0.699179 dB`;
- improve/regress/tie: **`80/20/0`** (`<=20` regressions, pass);
- worst paired PSNR: **`-3.578444 dB`** (`< -3.0`, fail);
- mean RGB-SSIM: **`+0.0065919`** (`>= +0.006`, pass).

Relative to exact T036, the fixed hybrid is slightly worse on mean quality (`-0.018580 dB` PSNR, `-0.001764` RGB-SSIM), so the preregistered classification remains: **`T059-E gain-direction advantage does not translate into a sufficiently safe/material fixed T036 trajectory improvement`**. No gate may be relaxed post hoc.

The safety shift remains informative: compared with T036's `29/100` regressions and `-5.614 dB` worst case, T060-C-R1 reduces regressions to `20/100` and improves the worst case to `-3.578 dB` while retaining nearly all mean PSNR benefit. This is a mechanism clue, not a promoted method.

### T060-D-R2 — source trajectory-vs-selector audit: negative; rescue line closed

The fixed source-only audit compared literal T036/T014 against literal T060-C-R1 on the same 60 T060-B source anchors. All primitive/downstream equivalence checks passed; both 41-state trajectories per anchor froze before any source-clean access; no target-development or official-test quantity entered adaptation or selection.

Paired B−A source results:

- selected PSNR: **`+0.413501 dB` mean**, `+0.220876 dB` median, `37/60` wins;
- oracle-best PSNR: **`+0.505451 dB` mean**, `+0.002797 dB` median, `35/60` wins;
- selection-regret difference: **`+0.091950 dB` mean**.

The fixed gates required oracle mean `>=+0.15`, oracle median `>0`, oracle wins `>=36/60`, and regret difference `>=+0.10`. The last two fail (`35/60`, `+0.09195`), so the accepted classification is **`T014-selection mismatch is not a sufficient explanation for the T060-C-R1 near-miss`**. The T059/T060 action-transfer rescue line is closed; no selector fitting or gate relaxation is authorized for that line.

A broader bottleneck signal survives independently of T059: absolute source selection regret is very large for both trajectories — T036/T014 `6.637364 dB` and T060-C-R1 `6.729313 dB`. Together with T037-A's development-only `0.683234 dB` mean reference-best headroom and `18/29` earlier-rescuable T036 PSNR-loss cases, checkpoint selection remains worth studying as a general T036 problem rather than as a T059 rescue mechanism.

## Development versus final-evaluation protocol

The fixed 100-image LOL-v2 Real cohort drawn from the training split is explicitly a **development set**. It may be used for predeclared method design, hyperparameter selection, ablations, and failure analysis. It must **not** be used to state the final Ours-vs-baseline performance gap, because strong released supervised baselines can be training-exposed to this split.

Final comparison rules:

- Freeze the final Ours method, model assets, action space, optimizer/stopping rule, and all hyperparameters before final held-out evaluation.
- The standard in-domain comparison must use the **complete official LOL-v2 Real test split** under the same frozen inference protocol. No test clean/normal target, PSNR/SSIM, label, reference gradient/Jacobian, or per-image outcome may enter adaptation or selection.
- A **cross-dataset / domain-shift held-out evaluation is required** to test the stated unknown-degradation motivation. Candidate complete held-out test splits include LSRW and UHD-LL (or equivalent fixed datasets), with the same frozen Ours checkpoint/rule and no target-specific retraining or tuning.
- Development baseline numbers are diagnostic anchors only. Final baseline-gap claims must come from complete held-out test sets with training exposure/protocol disclosed.
- T060-specific post-hoc gate relaxation on already-open results remains prohibited even though the 100-image cohort is development data.

The official LOL-v2 Real test and cross-dataset held-out test sets remain sealed until Final Ours is frozen.

## Strong baseline development anchors

- **Retinexformer T033-A:** `21.4787864 / 0.7900612`.
- **SNR-Aware T045-A:** `23.3963299 / 0.8237644`.

Both are target-free at inference under the frozen comparison protocol, but their released supervised checkpoints are exposed to LOL-v2 Real training data containing this development split. They are training-exposed development anchors, not independent held-out SOTA evidence and not valid final Ours-vs-baseline gap estimates.

The final sprint objective remains strong performance against the strongest fair target-free baselines on complete held-out protocols, without violating the no-test-target rule. Any `+2–3 dB` target is an objective, not a current claim.

## Information-boundary rules

- Test-time adaptation and checkpoint/state selection must never consume test labels, clean/normal-light targets, reference gradients/Jacobians, oracle values, PSNR/SSIM, degradation masks/gain maps, condition IDs, annotations, semantic image IDs, or per-image baseline outcome/harm labels.
- A degraded-image feature/action Jacobian such as `dx/dv`, `dx/du`, or `dx/d(raw_gain)`, computed entirely from the current degraded image/current target-free intermediate image and frozen model, is permissible; a Jacobian/gradient that depends on a clean/reference target is not.
- Development clean/reference targets may be used only offline for globally predeclared method development/evaluation and may never become per-image inference inputs or selectors.
- Source-training clean/reference targets may be used for source-supervised training or isolated source-domain diagnostics; they are never admissible test-time inputs.
- Reference diagnostics may motivate only global research choices; no per-image oracle quantity may enter deployable inference.
- External baselines admitted to final main comparisons must be target-free at inference, and their training exposure must be disclosed.
- Fresh/final benchmark sets must remain isolated from model/hyperparameter selection.
- **The official LOL-v2 Real test remains untouched through T060-D-R2.** Cross-dataset held-out sets remain sealed for final evaluation.
- Fresh/test runs must fail closed on source/checkpoint/cohort/provenance mismatches.

## Best current methods / ceilings

- **Broad fresh-qualified Ours-Core:** T014 Sobolev Region2 TTT.
- **Best fixed-validation deployable candidate:** T026-A, `11.1208764 / 0.3737918`.
- **Fresh-qualified deployable action extension:** T036-A common gain, `+0.9392571 dB` mean on its fresh cohort, unsafe tail unresolved (`29/100` regressions; worst `-5.614 dB`).
- **Best promoted non-deployable mechanism state:** T055-A, `24.3497922 / 0.7591279`.
- **T059 matched-detail:** direction transfer is scientifically supported, but practical detail gains are negligible; the integration line is closed.
- **T060-A spatial exposure:** full-field transfer failed.
- **T060-B common-gain direction:** positive first-order result; T059-E clearly out-aligns T014 in the exact 4-D gain subspace.
- **T060-C-R1 finite-step hybrid:** fixed negative by preregistered gate; preserves almost all T036 mean PSNR and improves tail count/severity, but misses the worst-regression safety gate and is not deployable.
- **T060-D-R2 selector diagnosis:** fixed negative for the T059 rescue hypothesis; T059/T060 action-transfer line closed. Large absolute source selection regret motivates a separate T036 selector investigation.

## Integration note

Historical PRs may contain inherited history/integration complications. Preserve accepted scientific/evidence states rather than repairing history inside experiment cycles. T060-C-R1 is reviewable through tested source `92c8bdd38e51b7d94c9a16cbd2e5c729dd382e52`, evidence `3963d36aac5d2b3d963b52b1b429020c88b09859`, and PR #128. T060-D-R2 is reviewable through tested source `2f83e7bda8b7be64f1a520d4ad3afc144ad503be`, evidence `23c98d16c159e8f51c5147647281ae749741dade`, and PR #131; do not spend experiment cycles repairing inherited PR history.

## Current open task

**T061-A — source-chosen global stopping-step transfer audit** in `coordination/CHATGPT_TO_CODEX.md`.

Reuse existing frozen source T036 trajectories and existing T037-A 100-image development trajectories only. Choose one global fixed step from source PSNR alone, freeze it before reading development per-step outcomes in this task, then test whether that same constant step improves T036 on development without per-image target information. No new optimizer run, learned selector, T059-E, baseline/SOTA comparison, official LOL-v2 test, or cross-dataset held-out access is authorized.