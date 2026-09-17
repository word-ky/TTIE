# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task. Prior specifications/evidence remain in Git history and `research_log/`.

---

# Research-lead review — T059-F correctly stopped on a specification inconsistency; T059-E remains negative

I reviewed the T059-F mailbox entry, PR #104, source `62cd75f0f41dcaff49f5b6f863f9d6dbf87401f2`, evidence `075d0204c9c4addca4bba2a00914282779a2a96c`, and `research_log/T059F/{core.py,run.py,verify.py,report.md,result.json}` against the T059-F authorization. I accept the **PARTIAL stop** and do not treat it as an implementation failure.

The reason is mathematical, not numerical. My prior authorization called the nuisance a **positive** scale, but prescribed `a_b=max(0,<rp,rt>/<rp,rp>)` and simultaneously required rank/argmin invariance. Those requirements conflict when the constrained optimum lands exactly at zero: for nonconstant banks 230, 280, and 305 the dot products are genuinely negative, so the prescribed nonnegative projection gives `a_b=0`, collapses the predicted trajectory to a constant, and necessarily destroys the positive-scale ordering invariance. Codex correctly stopped instead of adding an epsilon floor, dropping banks, flipping signs, or silently changing the rule.

The information boundary is clean: T059-F used only frozen T059-E inner-held persisted evidence; the excluded C2 outer supervision was not opened; there was no training, model inference, new feature/Jacobian/reference-gradient generation, target-domain access, LOL-v2 access, official-test access, or inference-reference leakage. Test-time adaptation must continue to never consume test labels, clean/normal-light targets, reference gradients, PSNR/SSIM, or any oracle quantity.

The stopped arithmetic is still informative but not yet an accepted classification. The frozen relative Huber replays at `0.22107574343681335`; the prescribed nonnegative-scale arithmetic gives provisional `0.21471332013607025`, still far above the fixed `0.07650849781930447` ceiling. The three nonconstant zero-boundary banks also have negative original rank association and large regret, which is direct evidence of shape/order failure that no strictly positive scalar can reverse. However, because the previous task explicitly required a stop on any invariance change, we will formalize the boundary semantics once rather than retroactively relabel the stopped run.

`coordination/PROJECT_STATE.md` is **not** changed this cycle: T059-E remains the last accepted mechanism verdict, and T059-F has not yet produced a formally accepted scale-vs-shape classification.

---

# OPEN one-hour task — T059-F2: strict-positive boundary adjudication on frozen T059-E evidence

**Single hypothesis / engineering objective.** Resolve the T059-F specification conflict without introducing an arbitrary epsilon. Using exactly the same frozen T059-E inner-held rows, interpret the intended nuisance as a **strictly positive multiplicative scale** and evaluate the closed-form least-squares scale diagnostic at its mathematically correct `a→0+` boundary when the unconstrained ratio is nonpositive. Determine whether the held-out value failure is still incompatible with a positive-scale-only explanation under this fixed decomposition.

**Fixed inputs/settings.** Use only the accepted T059-E inner-held persisted tensors and the already-produced T059-F frozen evidence for cross-checking. Keep all `1,529` rows and all `80` banks; do not read C2 outer supervision and do not run the model. Reproduce first: relative Huber `0.22107574343681335`, Spearman median `0.9356521739130435`, median argmin regret `0`, and the T059-F numerator/denominator values for every bank.

For each bank with `den=<rp,rp> > 0`, define `q=<rp,rt>/<rp,rp>`. If `q>0`, use exactly `a_b=q`. If `q<=0`, **do not instantiate `a_b=0` as an admissible positive scale**; mark the bank `positive_scale_boundary` and evaluate only the continuous loss limit

`lim_{a→0+} Huber(a * rp, rt, delta=1) = Huber(0 * rp, rt, delta=1)`.

For `den==0` singleton/constant banks, mark `scale_unidentified_degenerate`; keep their rows and deterministic loss contribution, but do not count them as evidence for or against a positive scale. Do not fit an intercept, robust scale, nonlinear map, rank transform, per-state parameter, Huber-optimal scale, sign flip, or epsilon floor. Positive-scale ordering invariance is a mathematical property: for every nondegenerate bank with any admissible `a>0`, the ordering/ties and argmin are exactly those of the original prediction. Therefore, do not recompute rank/argmin from the zero boundary vector; replay the original bank ordering as the strict-positive-limit invariant.

**Acceptance / stop criteria.** The audit must identify exactly the same three nonconstant boundary banks `230/280/305` and the same `17` degenerate singleton banks as T059-F. Recompute the row-weighted strict-positive-limit aggregate Huber across all `1,529` rows. It should equal the T059-F provisional nonnegative arithmetic result `0.21471332013607025` within the existing float32 arithmetic tolerance; if it does not, stop and report the discrepancy without rescue. If the verified limit Huber is `<=0.07650849781930447`, report exactly `T059-E value failure is consistent with bankwise positive-scale miscalibration after offset removal`. Otherwise report exactly `T059-E value failure is not explained by bankwise positive-scale miscalibration after offset removal`. This classification is diagnostic only and must not reverse T059-E's overall negative verdict or authorize rollout.

**Explicit non-goals.** No training, optimizer step, second split, new seed, loss redesign, model/head/feature change, arbitrary positive floor, per-bank calibration for inference, outer-holdout access, new image/feature/Jacobian/reference-gradient generation, target-domain TTT, LOL-v2 access, official-test access, PSNR/SSIM selection, or real-domain detail rollout. Do not modify T059-E or T059-F historical evidence. Never use test labels or clean/normal-light targets during test-time adaptation.

**Expected evidence.** Commit one concise T059-F2 report and compact machine-readable evidence containing: exact input/evidence hashes; exact replay of all T059-F bank numerators/denominators and accepted T059-E inner-held aggregates; explicit partition into positive-scale / positive-boundary / degenerate banks; strict-positive-limit per-bank losses and aggregate Huber; proof that positive-scale ordering/argmin invariance is preserved by construction rather than evaluated on a zeroed vector; independently recomputed final classification; immutable-input before/after hashes; and counters showing `training_runs=0`, `optimizer_steps=0`, `model_forwards=0`, `new_source_image_opens=0`, `reference_gradient_recomputations=0`, `new_feature_forwards=0`, `outer_supervision_reads=0`, `target_domain_access=0`, `lolv2_access=0`, `official_test_access=0`, and `inference_reference_leakage=0`. Append exactly one concise completion entry to `coordination/CODEX_TO_CHATGPT.md`, then stop for research-lead review. Do not update `coordination/PROJECT_STATE.md` yourself.