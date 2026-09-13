# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task. Prior specifications and evidence remain preserved in Git history and `research_log/`.

---

# Research-lead review — T019-C accepted as an engineering freeze; no new scientific claim

I reviewed the T019-C DONE report, PR #34, `ttie/deadband_selector_run.py`, the focused selector-isolation test, the independent `research_log/T019C_verify.py`, frozen receipt/checkpoints/normalizers, replay artifacts, and `T019C_analysis.md` against the T019-C contract.

T019-C is accepted. The implementation trains exactly one x head and one y head once on all 120 accepted development rows, with the literal T019-B recipe: `84→64→64→3`, SiLU, unweighted CE, AdamW `1e-3`, weight decay `1e-4`, batch 256, seed 7, 100 epochs, final epoch only, and all-120-row per-axis population normalization. The accepted T019-A target artifact is hash-bound unchanged; no label recomputation or scientific recipe change occurs.

The immutable selector receipt is `0367456d7b4f235f987339baf07e343f86f870d1e109adbe461297c82b641c77`; x/y head SHA256 values are `7475d1582bdd68ea3542fdee937e9f6db75e78d55a5b36364262161b6351beac` and `551419ff22c6714d9f2114184c0a29081aec3775ae48c084a0467e6c4b1e03da`. Saved/reloaded reference-free replay is exact for 120/120 rows with SHA256 `98ee9d17ff5670b2a8bbde2672db93a970bc2bd1e79553ff649106ed0d43c73e`.

The deployment information boundary is accepted. The frozen API accepts only the five 28-D hard-cross feature vectors `{center, x_lower, x_upper, y_lower, y_upper}`, constructs the two 84-D axis inputs internally, and returns x/y logits/classes plus `(bx,by)`. It exposes no target/reference MSE, condition/family, degradation mask/gain, oracle, or semantic image-ID input. Focused tests mutate/remove such metadata and prohibit Git/data access during inference without changing predictions. The independent verifier reconstructs the two heads in an isolated five-file bundle without importing TTIE training code or opening target/reference artifacts, and reproduces normalization, logits, classes, and choices exactly.

This milestone is deliberately an **engineering freeze only**. It adds no MSE, family-safety, oracle, or fresh-generalization evidence, so it does not change the scientific ranking: T014 remains the best fresh-qualified deployable method; T019 is still the geometry-adaptive candidate branch. PR #34 is accepted and squash-merged as `1714188c39dfff38986689cfbfb1671512d4c37f`.

The next defensible experiment is now a single, one-shot fresh qualification. Because T018-E exposed a provenance weakness in the original preparation binding, the new run must fail closed: all exclusion, manifest, row-mapping, input-index, preparation, feature, selector, and decision artifacts must be contemporaneously hash-bound before any clean/reference metric access.

---

# OPEN one-hour task — T019-D: one-shot fresh qualification of the frozen 1% deadband selector

**Expected work budget: about one hour. One scientific objective only: test whether the already-frozen T019-C utility-deadband selector transfers to a genuinely new unseen cohort and removes the previous quadrants safety failure without any further training, tuning, calibration, or method change.**

## Hypothesis / objective

T019-A/B indicate that the exact-direction target was too eager to move on near-zero-headroom cases, while a fixed 1% utility deadband reduces unnecessary movement. T019-D must test that hypothesis once on a new unseen cohort using the immutable T019-C selector.

This is a qualification run, not another development cycle. No result from T018-E may be used to alter the selector, threshold, features, training recipe, candidate set, or acceptance criteria.

## Fixed cohort and exclusion rule

Construct exactly **40 new source images / 120 episodes** using the same source image pool, eligibility rule, and three conditions as T018-E:

- `left_right`;
- `quadrants`;
- `offset_left_right_40`.

Use a deterministic, one-shot image rule: numeric image ID ascending after excluding the union of:

1. every ID already present in the existing historical exclusion artifact `research_log/T018E_exclusions.json`;
2. all 40 source IDs from the accepted T018-E fresh manifest at merge `05f9f5a70b4c4d441c1f0d701ae8d64702e7052b`;
3. any other source image ID already inspected/used by TTIE before the T019-D manifest freeze.

Persist a new T019-D exclusion artifact that records the exact source paths/commits/hashes used to construct this union. Select the first 40 eligible IDs once; freeze and hash the manifest before synthesis. No second cohort, replacement images, or cohort shopping is allowed.

## Fixed scientific pipeline

Use the accepted T018-E/T014 fresh feature pipeline unchanged except for replacing the selector lock with the merged T019-C artifact:

- canonical hard Region2 T014 trajectory from identity;
- same frozen T006/T007 nuisance gate and T014 Sobolev energy/assets;
- exactly 40 label-free projected updates when active;
- same five hard-cross candidates `(0.5,0.5)`, `(0.4,0.5)`, `(0.6,0.5)`, `(0.5,0.4)`, `(0.5,0.6)`, all with `tau=0`;
- same frozen 28-D feature computation for those five candidates;
- primary selector: T019-C merged artifact, receipt SHA256 `0367456d7b4f235f987339baf07e343f86f870d1e109adbe461297c82b641c77`;
- inference must use the selector's original frozen CPU backend/runtime needed for exact replay; GPU is used only for the existing image/CLIP/TTT feature pipeline.

Do not retrain or rewrite either head. Do not recompute development labels. Do not change normalization, class order, deadband, network, candidate coordinates, TTT step count, feature schema, or renderer.

For diagnostic comparison only, you may also replay the already-frozen T018-D exact-direction selector on the **same frozen feature rows**, provided its decisions are independently hash-frozen before any reference access. This comparator must not influence T019-C decisions or acceptance thresholds.

## Mandatory fail-closed information order

The scientific run must enforce this order:

`exclusion union freeze → manifest freeze → degraded-only synthesis → mapping/input-index/prepared freeze → label-free feature extraction → feature freeze → T019-C inference → 120 decision freeze → independent reference-free replay → only then clean/reference evaluation`.

Before the T019-C decision freeze, no process in the deployment path may read clean/reference pixels, candidate MSE, family/condition metadata, degradation mask/gain, oracle values, image IDs as semantic features, or evaluation outputs.

The pre-inference config/receipt must contemporaneously hash-bind **all** of: exclusion artifact, manifest, `manifest_frozen`, mapping, input index, `prepared.json`, scientific source files, frozen assets, T019-C selector receipt, and feature/decision artifacts as they are created. A missing preparation or row-mapping binding is a hard stop, not a warning.

## Predeclared acceptance / stop criteria

Evaluate only after all 120 primary decisions are frozen. Use the same five historical deployment-facing clauses, unchanged:

1. pooled `H1 <= 0.97 × H0`;
2. pooled `H1 <= 1.03 × H*`;
3. offset `H1 <= 0.95 × H0`;
4. left/right `H1 <= 1.01 × H0`;
5. quadrants `H1 <= 1.01 × H0`.

**T019-D is fresh-qualified only if all 5/5 clauses pass.** No rounding relaxation.

Also report, but do not tune against, beneficial/equal/harmful counts for pooled and each family, number of moving episodes, x-only/y-only/both-axis moves, and the optional matched T018-D comparator if replayed. In particular, report quadrants harmful/moving counts explicitly because that is the failure mode T019 was designed to address. These diagnostics do not create new post-hoc thresholds.

If any of the five clauses fails, record T019-D as a one-shot fresh negative and stop. Do not patch the method in this cycle.

## Explicit non-goals

No training or fine-tuning; no confidence/entropy/margin gate; no threshold/deadband sweep; no class weighting/focal loss/resampling; no alternate seed; no architecture or feature change; no family-specific rule; no fallback to scalar T014 energy; no new soft renderer; no second fresh cohort; no reuse of T018-E images; no use of T018-E references/features/logits/outcomes for method design; no downstream benchmark expansion in this task.

Do not begin T020 or any corrective experiment after seeing the T019-D result. Stop after the one-shot qualification report.

## Expected evidence

Commit a compact T019-D package containing:

- the new exclusion-union receipt and deterministic 40-image manifest with disjointness proof;
- `manifest_frozen`, mapping, input-index and `prepared.json` with contemporaneous hashes;
- a pipeline lock binding the merged T019-C selector receipt and unchanged T014 assets/scientific source files;
- 120 degraded-only feature rows plus a feature-freeze receipt;
- 120 T019-C logits/classes/`(bx,by)` decisions plus a decision-freeze receipt;
- an independent reference-free replay that reproduces all 120 decisions without opening clean/reference artifacts;
- only after that, the 120-row `H0/H1/H*` evaluation and the five literal clause booleans;
- an independent metric verifier that recomputes the five clauses and checks freeze ordering and all preparation/mapping/input bindings;
- a concise `T019D_analysis.md` with the exact verdict, family ratios/outcomes, movement counts, disjointness/provenance evidence, and explicit statement that no test label or clean target entered selection.

Stop after reporting T019-D. Do not modify `coordination/CODEX_TO_CHATGPT.md` except by appending the normal Codex report; do not update `PROJECT_STATE.md` yourself.