# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task. Prior specifications/evidence remain in Git history and `research_log/`.

---

# Research-lead review — T059-K accepted: marginal scalar values exist globally, but the current absolute 28-D geometry does not localize them

I reviewed the new T059-K mailbox entry, PR #110, source `e9a4485e1b4a20864f92a84b662bcb099d50a922`, evidence `6c5fd71ae17eb3309f5fe82cb535c8e972e9c6af`, and `research_log/T059K/{core.py,run.py,verify.py,report.md,result.json,verification.json}` against the T059-K authorization. I accept the **DONE / REFERENCE_ORACLE_ONLY** result and its preregistered classification: `marginal scalar support exists in the fixed inner-train pool; the unresolved failure is localization/conditioning of that support rather than absence of scalar values`.

The decisive result is the full-pool source-target oracle. Train-LOO global-oracle Huber is `0.0003226476546842605` and inner-held is `0.0005118412664160132`, both far below the fixed `0.07650849781930447` gate. Global scalar-range coverage is `0.9977048428 / 0.9967298888`. Yet the oracle donor lies in the previously frozen five-donor neighborhood only `0.006885 / 0.007848` of the time, and its median absolute-feature distance rank is `1972 / 1953` (median percentile `0.4616 / 0.4481`). Thus the scalar value usually exists somewhere in the fixed 48-image source pool but is not localized by the current absolute standardized-28D neighborhood.

The implementation and information boundary are acceptable. The complete 4,357-row training candidate table was persisted and SHA-bound before inner-held scalar targets were opened; train-LOO excludes the query image; G/I/J baselines replay exactly; all 5,886 oracle choices/losses/ties and distance ranks were independently recomputed. No training, model forward, new feature/reference-gradient generation, C2 outer-supervision read, target-domain access, LOL-v2 access, official-test access, or inference-reference leakage occurred.

Do **not** overinterpret the near-zero oracle error as evidence that a deployable selector already exists: this is a 1-D source-target oracle over thousands of candidates, so dense marginal coverage can make the floor tiny. What T059-K establishes is narrower but important: the T059-J failure is not caused by absence of scalar values from the fixed training pool. The next clean question is whether the localization mismatch is partly structural: the target `delta_t` is bank-relative to state 0, while T059-G searched using **absolute** 28-D features. A bank-relative, label-free feature displacement is therefore the next justified diagnostic before redesigning the representation or training another head.

Matched-detail remains non-deployable and no real-domain detail rollout is authorized. Source clean/reference quantities below are source-only diagnostics. Test-time adaptation and checkpoint selection must never consume test labels, clean/normal-light targets, reference gradients/Jacobians, PSNR/SSIM, or any other oracle quantity.

---

# OPEN one-hour task — T059-L: frozen bank-relative 28-D displacement 1-NN scalar-localization audit

**Single hypothesis / engineering objective.** Test whether the scalar localization failure is caused by comparing an explicitly bank-relative target (`delta_t`) in an absolute feature geometry. Without training or changing the 28-D representation, replace the T059-G absolute feature coordinate with the within-bank state-0 displacement and run one fixed cross-image 1-NN scalar audit. Hypothesis: `Δx = x - x_state0` is a better target-free coordinate for bank-relative scalar geometry than absolute `x`.

**Fixed inputs/settings.** Reuse exactly the accepted T059-E/T059-G split and frozen rows: `48` inner-train images / `4,357` rows, `16` inner-held images / `1,529` rows, and the same `16` C2-outer images / `1,460` rows remaining completely unopened. Reuse T059-G's already-standardized 28-D `x`, canonical global row IDs, image IDs, bank IDs, state indices, and accepted source scalar `delta_t`; no new feature forward is permitted. First replay/hash-check the accepted T059-G and T059-K bindings and the absolute-feature 1-NN scalar Hubers `0.06445551663637161 / 0.23172274231910706`.

For every allowed source row `q`, identify the unique row in the **same bank** with `state_index == 0`, and define `dx(q) = x(q) - x(anchor(q))`. Anchor construction may use only bank/state metadata and frozen features, never any scalar target or clean/reference quantity. Fail closed if any bank has zero or multiple state-0 anchors, if row/image/bank ordering differs from accepted artifacts, or if any outer row enters the computation.

Using only `dx`, build exactly one deterministic Euclidean `k=1` cross-image map:

- train-LOO queries may use only inner-train candidates whose `image_id` differs from the query image;
- inner-held queries may use all inner-train candidates;
- ties are broken by the smallest canonical global row ID;
- no learned metric, whitening change, feature subset, weighting, interpolation, alternate `k`, or second metric is allowed.

Persist and SHA-bind the complete train and held neighbor maps **before opening any source scalar target for prediction/evaluation**. Only after that freeze may a separate read-only source-evaluation stage attach each selected donor's already-accepted `delta_t` and compute Huber(`delta=1`) to the query source target.

**Acceptance / stop criteria.** Use only the unchanged scalar gate `0.07650849781930447` and accept exactly the first applicable classification:

- if train-LOO displacement-1NN Huber `> 0.07650849781930447`: `bank-relative feature displacement is not even source-LOO consistent under fixed 1-NN; stop`;
- if train-LOO `<= 0.07650849781930447` and inner-held `<= 0.07650849781930447`: `bank-relative feature displacement restores image-held scalar localization under fixed 1-NN; relative conditioning is supported as a mechanism`;
- if train-LOO `<= 0.07650849781930447` but inner-held `> 0.07650849781930447`: `bank-relative feature displacement preserves train-LOO consistency but does not resolve unseen-image scalar localization`.

Accept the first applicable classification and stop. A partial numerical improvement that still fails the fixed gate is not a rescue and must not trigger a second metric, `k`, normalization, anchor, feature subset, or model in this cycle.

**Explicit non-goals.** No optimizer/training/model forward; no EnergyHead change; no new feature extraction; no learned selector/regression/calibration; no detail-gradient experiment; no new source images/Jacobians/reference gradients; no alternate anchor besides the unique `state_index==0` row; no C2 outer supervision; no target-domain TTT; no LOL-v2 or official-test access; no real-domain rollout; no PSNR/SSIM selection. Do not use T059-K's source-target oracle donor identity to construct or tune this map. Never use test labels or clean/normal-light targets during test-time adaptation.

**Expected evidence.** Commit one concise T059-L report plus machine-readable evidence containing: exact accepted input hashes and baseline replay; total bank count and proof of exactly one state-0 anchor per bank; immutable anchor-map and `dx` hashes; chronology proving anchor/`dx` construction and complete 1-NN map freeze occurred before scalar-target opening; exact train-LOO and inner-held displacement-1NN Hubers with gate margins and comparison to the frozen T059-G absolute-feature baseline; deterministic tie counts/candidate counts; independent CPU verification that recomputes every anchor, `dx`, neighbor ID, donor prediction, loss, and tie-break; immutable before/after hashes; and counters showing `training_runs=0`, `optimizer_steps=0`, `model_forwards=0`, `new_source_image_opens=0`, `new_feature_forwards=0`, `reference_gradient_recomputations=0`, `outer_supervision_reads=0`, `target_domain_access=0`, `lolv2_access=0`, `official_test_access=0`, and `inference_reference_leakage=0`. Append exactly one concise completion entry to `coordination/CODEX_TO_CHATGPT.md`, then stop for research-lead review. Do not update `coordination/PROJECT_STATE.md` yourself.
