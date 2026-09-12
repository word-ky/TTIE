# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task. Prior task specifications and evidence remain preserved in Git history and `research_log/`.

---

# Research-lead review — T014 accepted; PR #14 merged

PR #14 is accepted and squash-merged as `cebecffbd1335fade336df17d653eb4e5fb65ba3`.

T014 is a genuine fresh positive result under the frozen synthetic restoration protocol. The same 28-feature / same-MLP matched-control experiment supports the narrow causal claim that **explicit source derivative supervision materially improves the learned inner objective's test-time gradient geometry**. Stage A passed 8/8 development clauses, and after the required Git freeze barrier the immutable Stage B passed all 12 fresh clauses on 40 unseen COCO images. No fresh refitting, retuning, second split, or outcome-dependent repair occurred.

Key fresh results for `region2_ttt_energy_sobolev`:

- clean mean MSE `0.0002532601 <= 0.003`;
- clean p95 MSE `0.0001256009 <= 0.005`;
- dark/identity `0.57095 <= 0.60`;
- bright/identity `0.42042 <= 0.60`;
- heterogeneous/global `0.56250 <= 0.85`;
- heterogeneous/direct `0.84696 <= 0.95`;
- heterogeneous/discrete `0.88275 <= 0.95`;
- heterogeneous/fixed16 `0.91688 <= 0.95`;
- heterogeneous/value-only `0.80830 <= 0.95`;
- dark-region/identity `0.53157 <= 1.05`;
- bright-region/identity `0.44764 <= 1.05`;
- exact-quadrant/bilinear `0.71437 <= 0.90`.

The heterogeneous MSE is `0.03385803`, 19.17% below the same-source value-only control, 11.72% below matched projected discrete, and 8.31% below frozen semantic step16. The reference-only trajectory oracle is only 2.80% better than the deployed minimum-energy selector, so the positive result is not hiding a large checkpoint-selection failure.

The mechanistic evidence is also clean: Sobolev has *worse* source scalar Huber than value-only (`0.05101` vs `0.03319`) but far better unseen calibration gradient alignment (`73/74`, median `0.93606` vs `59/74`, median `0.42464`) and better fresh restoration. This is the strongest evidence so far that the useful quantity is the learned derivative field, not scalar value fit alone.

Two limitations remain binding and motivate the next experiment. First, the result is still **Sobolev objective + projected action geometry**: about 94% of primary updates are projected and roughly 63% of movable final coordinates end on a trust-box boundary. Second, hard `region2` remains basis-limited: on `offset_left_right_40` it is slightly worse than bilinear (`1.00589×`), while the global Sobolev control remains better on homogeneous dark/bright. Do not claim universal spatial-basis superiority or natural-degradation benchmark qualification from T014.

The non-negotiable rule remains: **test-time adaptation must never consume test labels, clean targets, degradation masks/gain maps, condition IDs, annotations, image IDs as semantic shortcuts, source-only Jacobians/reference gradients, or evaluation metrics.** Clean references remain post-persistence evaluation only.

---

# T015 — Frozen Sobolev Cross-Basis Routing

**Status: OPEN.**

## Scientific question

T014 established a useful learned derivative field, but no single fixed spatial basis dominates: global is better for homogeneous shifts, hard `region2` is strong on exact quadrant geometry, and bilinear is slightly better on the boundary-offset stress.

Test one minimal hypothesis before introducing a learned basis:

> Can the already-frozen T014 Sobolev energy itself choose the appropriate **basis** per test image, using no new training and no reference information?

This is a routing audit over already-qualified frozen trajectories, not a new energy, feature expansion, learned mask, detector objective, meta-initialization, or ViT3 model.

## Frozen ingredients

Reuse unchanged from accepted T014:

- T006 CLIP exposure readout and T007 gate/calibration;
- frozen T014 Sobolev head/checkpoint/normalization and 28-feature schema;
- global, bilinear2 and hard `region2` projected action geometries;
- identity reset, Adam `lr=0.03`, exactly 40 projected updates when active;
- per-basis checkpoint selection by minimum predicted Sobolev energy, earliest tie;
- direct, projected-discrete, semantic fixed16, and identity controls;
- all action boxes, gate logic, ISP operators and degradation definitions.

No source retraining, calibration stage, loss change, score normalization, basis-specific offset, temperature, learned router, or threshold is allowed in T015.

## Routing rule

For each input, independently run the frozen T014 Sobolev trajectories for:

- `global_ttt_energy_sobolev`;
- `bilinear2_ttt_energy_sobolev`;
- `region2_ttt_energy_sobolev`.

Each basis first chooses its own checkpoint exactly as T014 did. Then define the routed method by the **lowest frozen Sobolev energy value among the three selected checkpoints**. Use deterministic tie order `global -> bilinear2 -> region2` (prefer the lower-complexity basis on exact ties). No clean-reference metric, condition ID, gate-derived synthetic label, mask/gain, or annotation may enter this routing decision.

Persist all three basis trajectories, their selected checkpoints/energy scores, and the final routing decision/hash before any reference metric is attached.

Also compute after persistence a reference-only `oracle_best_basis` selecting the lowest-MSE output among the same three selected basis outputs. This oracle is diagnostic only.

## Fresh-only evaluation split

T015 has **no fitting or calibration**, so use one new deterministic fresh split only.

All 648 image IDs inspected through T014 are unavailable. From official COCO val2017 image files only, no annotations, numeric ascending, original shorter side >=320, take the next 40 eligible IDs as `evaluation_t015`. Commit the manifest and hash before running any output/reference metric.

Run six predeclared conditions on all 40 images:

- `clean`;
- `homogeneous_dark`;
- `homogeneous_bright`;
- `left_right`;
- `quadrants`;
- `offset_left_right_40`.

Unlike T014, `offset_left_right_40` is **primary** in T015 because boundary robustness is the scientific question.

## Required methods

At minimum persist/evaluate:

1. `identity`;
2. `region2_direct`;
3. `region2_discrete_projected`;
4. `fixed_step_source` (semantic step16);
5. frozen `global_ttt_energy_sobolev`;
6. frozen `bilinear2_ttt_energy_sobolev`;
7. frozen `region2_ttt_energy_sobolev`;
8. primary `routed_ttt_energy_sobolev`;
9. reference-only `oracle_best_basis` after persistence.

Do not choose or retrain any method from fresh outcomes.

## Predeclared qualification

Define `spatial_pool = left_right + quadrants + offset_left_right_40`. Define `best_fixed_basis_spatial` as the lowest aggregate reference MSE among the three frozen fixed bases **for evaluation only**; it is not available to the router.

`routed_ttt_energy_sobolev` qualifies only if all clauses hold:

1. clean mean MSE `<= 0.003`;
2. clean p95 MSE `<= 0.005`;
3. homogeneous dark MSE `<= 0.60 × identity`;
4. homogeneous bright MSE `<= 0.60 × identity`;
5. `spatial_pool` MSE `<= 0.97 × best_fixed_basis_spatial`;
6. `spatial_pool` MSE `<= 0.95 × region2_discrete_projected`;
7. `spatial_pool` MSE `<= 0.95 × fixed_step_source`;
8. offset MSE `<= 1.01 × bilinear2_ttt_energy_sobolev`;
9. aligned heterogeneous (`left_right + quadrants`) MSE `<= 1.01 × region2_ttt_energy_sobolev`;
10. routed `spatial_pool` MSE `<= 1.05 × oracle_best_basis`.

The 1% per-family clauses are non-inferiority checks; the 3% aggregate best-fixed-basis clause is the actual adaptive-basis value claim. Report all basis-selection counts by condition, the full energy-score margins between winner and runner-up, oracle basis counts, routing-vs-oracle disagreement rate, and regret conditioned on each selected basis.

## Interpretation rules

- If all ten pass, conclude only that the **frozen Sobolev energy is sufficiently cross-basis calibrated to route among three fixed bases** on this synthetic protocol. Do not call it a learned spatial basis.
- If the aggregate clause fails but oracle_best_basis is strong, the bottleneck is cross-basis energy calibration/routing rather than basis capacity.
- If the oracle itself cannot beat the best fixed basis by the required aggregate margin, routing is not worth learning; a genuinely more expressive spatial basis becomes the next candidate.
- If offset remains weak even for the oracle, hard/bilinear/global basis capacity is the limiting factor.

Preserve the negative result if any clause fails. Do not tune on this split, add score calibration, retrain the energy, or start a learned basis automatically.

## Required evidence

Add focused tests proving:

- router inputs contain only the three frozen label-free selected energy scores;
- replacement clean reference / metadata cannot change any trajectory, selected checkpoint or basis decision;
- tie order is literal and deterministic;
- per-basis T014 behavior is bitwise/regression-equivalent to accepted T014 on fixtures;
- outputs/trajectories/scores/routing decision/hashes are persisted before reference access;
- oracle_best_basis is evaluation-only.

Run full local and A6000 tests. Freeze scientific code and the 40-image manifest before outcomes. If T015 completes, stop and report; do not automatically start learned basis, detector, meta-learning, prompt retraining, or ViT3 work.

---

# Research-lead interim review — T015 frozen implementation accepted for fresh preparation

Reviewed Codex partial report `38ed3971` and frozen engineering commit `c4e58e5ad64bfce0bea72561997db8007e12b510` / draft PR #15 against the T015 contract and current project state.

The implementation is scientifically consistent with the frozen routing audit. `route()` receives exactly three scalar selected Sobolev-energy values and applies literal deterministic tie order `global -> bilinear2 -> region2`. `run_label_free()` reuses the unchanged T014 per-basis trajectory/checkpoint machinery; there is no new training, calibration, normalization, score offset, temperature, threshold, or learned router. The routed output is a copy of the already-selected frozen-basis result, so routing does not create a fourth trajectory.

The leakage boundary is also correct: all three basis trajectories, selected energy scores, outputs/decisions and `routing.json` are written and hashed first; only then does the reference callable enter `evaluate_episode()`. `oracle_best_basis` is computed from reference MSE only after persistence and cannot influence the router. The ten-clause reducer matches the predeclared contract, including primary `offset_left_right_40`, evaluation-only `best_fixed_basis_spatial`, the 3% adaptive-basis clause, both 1% family non-inferiority clauses, and the 5% oracle-regret bound. Current local evidence (132 full tests plus focused routing/persistence/fixture tests) is sufficient to proceed to the one frozen fresh experiment; PR #15 must remain draft until that experiment and evidence verification finish.

**Proceed, but preserve a strict freeze barrier before any T015 outcome is produced:**

1. Generate `research_log/T015_manifest.json` by metadata only from official COCO val2017 image files. Verify exactly 40 `evaluation_t015` IDs, exclusion of all 648 previously inspected IDs, original shorter side >=320, and no annotation access. Commit the manifest and record its SHA256 before launching the experiment.
2. Treat `c4e58e5ad64bfce0bea72561997db8007e12b510` as the immutable scientific implementation. The later manifest/evidence commit may add the manifest and bookkeeping only. Before launch, record `git diff --name-only c4e58e5..RUN_COMMIT` and verify there are **no changes** to `ttie/routing/**`, the frozen T014 donor code, `scripts/prepare_t015.py`, or `scripts/run_t015_a6000.sh`. Also preserve the existing T014 receipt/code-inventory verification. Do not repair or refactor scientific code after any fresh output exists.
3. Run exactly one formal 40-image × 6-condition = **240-input** A6000 evaluation with the frozen Sobolev checkpoint and the committed manifest. No duplicate run, score calibration, basis-specific correction, threshold tuning, or fresh-split replacement is permitted based on outcomes. The non-negotiable rule remains: **test-time adaptation/routing must never use test labels, clean targets, condition IDs, masks/gains, annotations, image IDs as shortcuts, source-only Jacobians/reference gradients, or evaluation metrics.**
4. At completion, verify all 240 episode receipts/hashes and recompute the summary independently from saved metrics. Report all ten clauses literally, the reference-only best fixed basis, `oracle_best_basis / best_fixed_basis_spatial`, basis-selection counts by each condition, oracle basis counts, full winner–runner-up energy-margin distribution, routing/oracle disagreement, and regret conditioned on routed basis. Preserve zero-oracle cases as absolute regret rather than silently dividing by zero.
5. If any clause fails, preserve the bounded fresh negative result and stop. In particular, do not learn a score calibration or a new spatial basis on `evaluation_t015`. If all ten pass, stop and report the positive routing result; do not start detector/meta/ViT3/learned-basis work automatically.

No `PROJECT_STATE.md` change is warranted yet: T015 has a reviewed frozen implementation but no fresh scientific outcome.