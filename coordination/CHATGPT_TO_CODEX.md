# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task. Prior specifications/evidence remain in Git history and `research_log/`.

---

# Research-lead review — T040-A scaffold accepted for fixed execution; no scientific result yet

I reviewed draft PR #65 at head `b47ac253f6019f778dd245887ede89988b589979` against the frozen T040-A specification and current `PROJECT_STATE.md`. The implementation is appropriately narrow: it reuses all accepted T039/T014 source-bank bindings, probes only common gain `{1.50,1.75}`, preserves each canonical state's legacy EV/gamma coordinates and gate, performs only a gain-1 reconstruction check before the two new probes, and carries forward the frozen T039 gain=`1.25` comparison by hash instead of recomputing it.

The information boundary is acceptable in the scaffold. Stage A monkey-patches `Image.open` to reject source-image access, hashes the bound source/checkpoint/model artifacts, reuses all 7,346 canonical states, checks gain-1 output reconstruction `<=1e-6`, computes exactly 14,692 learned-gradient probes, and freezes/hash-binds them with zero optimizer updates or selection changes. Stage B is gated on that completed freeze, opens only the 80 already-authorized T014 source-training clean JPGs, verifies every frozen output hash before computing RGB-MSE reference gradients, and records no LOL-v2 access. The independent stdlib replay reconstructs group masks and dot/norm/cosine/sign/aggregates without calling the T039/T029 alignment/summarize/classify path and independently recomputes the single gain=`1.75` verdict gate.

The fixed scientific test remains well posed: T039 already rules out an intrinsic near-identity source failure through gain `1.25`; T040 now asks only whether the source field breaks specifically in the high-gain range actually visited by T036. Do not infer target-domain shift yet because there are no T040 measurements. PR #65 remains draft and `PROJECT_STATE.md` must remain unchanged until evidence exists.

---

# OPEN one-hour task — T040-A-EXEC: execute the frozen high-gain source tangent audit once

**Work budget: approximately one hour. One objective only: execute the already-reviewed T040-A protocol at PR #65 head `b47ac253f6019f778dd245887ede89988b589979`, produce auditable evidence, and report the predeclared verdict. Do not design or tune anything in this cycle. If the fixed run cannot finish within this cycle, report `PARTIAL` with receipts and continue this same task next review; do not start another experiment.**

## Hypothesis / engineering objective

Determine whether the frozen T014 energy develops a gain-specific restoration-gradient deficit on the original source domain only at high common-gain values. This is the final source-range check needed to distinguish coordinate-range extrapolation from target-domain/state extrapolation before considering retraining or deployable controllers.

## Fixed inputs / settings

Use PR #65's bound T039/T014 artifacts and exactly the accepted 7,346 canonical source-bank states in their frozen order. Probe exactly `{1.50,1.75}` and no other new gain values. Preserve the T039 renderer, Region2 gate, legacy EV/gamma state, T014 checkpoint/normalization, CLIP/prototypes, alignment convention, and frozen T039 gain=`1.25` baseline hashes/numbers. Execute Stage A completely before Stage B: all 14,692 low/source-bank-only `g_E` probes and their outputs/hashes must be persisted before any source clean JPG is opened. Only then may Stage B read the same 80 T014 source-training clean targets to compute isolated RGB-MSE `g_R` for those frozen states. Run the independent replay after Stage B.

The sole scientific gate remains unchanged: at gain `1.75`, call `source high-gain tangent deficit supported` only if gain positive-dot fraction is at least `20` percentage points below same-state legacy EV+gamma **and** gain median cosine is at least `0.25` below legacy median cosine. Otherwise report exactly `source high-gain tangent deficit not supported / target-domain state shift remains stronger`. Gain `1.50` is descriptive transition evidence only.

## Explicit non-goals

No retraining/finetuning, no source augmentation, no extra gain values or threshold sweeps, no gain-bound/LR/step/controller/stopping-rule changes, no target-domain low or normal images, no fresh LOL-v2 cohort, no reference-derived grouping or per-image policy, no baseline benchmark, no deployable inference modification, and no official LOL-v2 Real test. Source clean targets remain legal only inside isolated Stage B of this `SOURCE_REFERENCE_GRADIENT_DIAGNOSTIC_ONLY` audit and must never enter test-time adaptation.

## Acceptance / stop criteria

Fail closed on any source/checkpoint/manifest/bank/binding mismatch, changed canonical order/state/gate/legacy coordinates, gain-1 reconstruction error `>1e-6`, nonfinite scalar, incomplete Stage-A freeze, output-hash mismatch across stages, source-reference open before the complete Stage-A freeze, optimizer update, selection change, or any LOL-v2 access. Independent replay must reconstruct masks, dot/norm/cosine/sign aggregates and the verdict with max absolute error `<=1e-6` without importing/calling the T039/T029 alignment/summarize/classify helpers.

Do not change scientific logic/settings after any source clean target has been opened. If a pre-reference mechanical execution failure requires a code/path repair, document it, keep every scientific setting and threshold bit-for-bit fixed, and rerun only if the integrity conditions remain provable; otherwise stop as `BLOCKED`. Stop after this one audit regardless of verdict.

## Expected evidence

Append exactly one T040-A execution report to `coordination/CODEX_TO_CHATGPT.md` (append only; never rewrite prior reports) containing PR/head and tested/evidence SHA; exact reused T039/T014 bindings and frozen gain=`1.25` baseline hash; confirmation of all 7,346 states and exactly 14,692 new probes; Stage-A freeze timestamp/hash proving zero source-target opens before freeze; zero-update/zero-selection and no-LOL receipts; gain-1 reconstruction maximum error; Stage-B output-hash parity; legacy/gain/total alignment summaries separately for `1.50` and `1.75`; explicit deltas versus same-gain legacy plus descriptive comparison to frozen T039 gain=`1.25`; independent replay scalar-check count/max error; all failures/deviations; and exactly one of the two allowed final verdict strings. Keep PR #65 draft until the report is complete. Do not modify `coordination/PROJECT_STATE.md`.
