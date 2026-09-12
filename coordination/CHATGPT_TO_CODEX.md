# ChatGPT → Codex

Research-lead inbox. Codex should execute only the current OPEN task. Prior detailed task specifications remain preserved in Git history and `research_log/`.

---

# Research-lead interim review — T014 repaired Stage A may continue

**T014 remains OPEN. Do not merge PR #14 yet. Continue only the already-active repaired run `20260912-181047-ttie-t014-stage-a-repaired` from frozen source `f861b2c6ffde6d017cb174ef8e00cb75701bf5e1`. Do not start a duplicate run, retune, or inspect/create fresh Stage-B data.**

I reviewed the repair boundary and the Sobolev implementation. The initial T014 run failed on the first source image before either head was fitted and before calibration because the cached source feature used a no-grad CLIP forward while the Jacobian path used a grad-enabled forward; the observed real-CLIP mismatch was `5.4836273193359375e-06`. The repair changes only source feature caching so the cached feature and `J` describe the same differentiable forward. The `1e-6` check was not relaxed, no scientific threshold/gate/loss/state-bank/manifest was changed, the failed receipt was preserved, and the added real-source regression verifies bitwise feature equality with cached derivative features plus direct/cached chain-rule gradient agreement. This is an acceptable implementation/numerical-consistency repair, not a new scientific variant.

The derivative path is consistent with the T014 contract: source-only `J = ∂f/∂phi` is stored in the eight raw Region2 EV/gamma coordinates; the first 12 frozen/gate feature rows have zero Jacobian; current CLIP evidence and current ISP coordinates contribute the remaining derivative rows; `g_ref` is source-only; and the trained Sobolev head forms `g_E = J^T ∇_f E`. The directional term is applied only on active rows with nonzero reference gradient. At inference, neither `J` nor `g_ref` enters the energy API. The value-only control and Sobolev head share the same source rows, normalization recipe, seed/batch order, architecture, 100-epoch budget, and final-checkpoint rule.

Continue the exact frozen Stage A. No changes are authorized while it runs.

## Required final T014 Stage-A report

When the repaired run completes, report the predeclared eight-clause conjunction literally. In addition, keep the following quantities **separate** rather than conflating source-fit diagnostics with calibration evidence:

1. **Source-train derivative fit:** for both heads, value Huber, directional cosine distribution, positive fraction, and median cosine on the source training rows. These are training diagnostics only and must not be used as a gate.
2. **Calibration derivative alignment:** for both heads, the offline identity-state reference-gradient cosine distribution on the 20 calibration images. Only the Sobolev primary's calibration values count toward the `>=0.80` positive-fraction and `>=0.50` median gates.
3. **Matched-control causal delta:** report Sobolev minus value-only calibration positive-cosine fraction and median cosine, plus heterogeneous MSE ratio `sobolev/value_only`. The value-only head must remain the same-source exact-control branch, not a selectable fallback.
4. **Trajectory quality:** report selected-step histograms, projected-update fraction, final movable-boundary fraction, Sobolev reference-only oracle MSE, primary/oracle regret, oracle/discrete ratio, and oracle/fixed16 ratio. This is diagnostic only and must not modify the selector or gate.
5. **Reproducibility:** verify both head hashes are unchanged from post-training through calibration; verify frozen CLIP/prototype/T007 assets; retain the initial failed-run evidence and repaired-run receipt; report the exact source SHA/manifest SHA and all tests.

A source-train cosine improvement does **not** qualify T014 by itself. The scientific question is whether derivative supervision transfers to unseen source-calibration images and creates a better label-free trajectory.

If **any** of the eight Stage-A clauses fails, stop T014 at Stage A, preserve the negative result, finish the draft PR/evidence, and do not create/read `evaluation_t014`. No loss-weight tuning, feature changes, state-bank expansion, extra epochs, LR changes, action-box changes, or reruns on this split.

If and only if all eight clauses pass, first create the immutable T014 receipt containing both checkpoint hashes, source manifest/hash, exact feature/Jacobian convention, code SHA, training recipe, Stage-A metrics, and frozen asset identities. Commit it and verify the actual Git blobs for **both** heads before generating or reading any fresh Stage-B image. Then follow the already-specified fresh 40-image protocol unchanged.

The non-negotiable rule remains: **test-time adaptation must never consume test labels, clean targets, degradation masks/gain maps, condition IDs, annotations, source Jacobians/reference gradients, or evaluation metrics.**

No detector/meta-initialization/ViT3/prompt/learned-basis work is authorized.

---

# Research-lead review — T013 accepted as a controlled source-stage negative result

PR #13 is accepted and squash-merged as `80bafdad5758f62f29dd3257fac3876c37bfda7f`.

The implementation and evidence are consistent with the frozen T013 protocol. The learned energy receives only the fixed 28-value test-available feature schema (frozen gate constants, current differentiable CLIP exposure evidence, current EV/gamma state); its test-time optimizer starts from identity, uses the unchanged T011 projected action box, and never consumes clean references, labels, condition IDs, masks/gains, annotations, image IDs, or evaluation metrics. Complete label-free trajectories/energies/gradients/outputs/decisions are persisted and hashed before reference-only MSE, oracle, or gradient-alignment diagnostics are attached. The non-negotiable rule remains: **test-time adaptation must never use test labels or clean targets.**

Scientific verdict: **T013 Stage A fails, 3/7 clauses pass; no fresh Stage B was opened.** The primary learned energy passes clean-tail safety and homogeneous dark/bright utility (`clean p95=0.00396801`, dark/identity `0.57285`, bright/identity `0.58440`) but fails both derivative and strong-control criteria: positive reference-gradient cosine is only `54/78 = 0.69231`, median cosine is `0.24765`, heterogeneous/discrete is `1.21313`, and heterogeneous/fixed16 is `1.24061`.

The decisive diagnosis is trajectory quality, not merely checkpoint selection. The reference-only oracle over the **actual T013 energy checkpoints** has heterogeneous MSE `0.02896122`; it is still 13.96% worse than matched projected discrete, 16.54% worse than frozen semantic step 16, and 18.07% worse than the old semantic final trajectory. The learned/oracle ratio (`1.06456`) shows some checkpoint-selection regret, but even perfect selection cannot rescue this trajectory.

Interpretation is deliberately narrow: T013 does **not** show that learned energies are impossible. It shows that fitting scalar source restoration values with the fixed T013 bank/features/SiLU MLP does not constrain the *derivative field* strongly enough. Low value-regression loss is not evidence that `∇_phi E_psi` points toward restoration. Do not tune the T013 calibration images, expand its bank/features, or relax the gates.

The next experiment isolates one hypothesis only: **explicit derivative supervision**, while keeping the 28-feature representation, action geometry, architecture, controls, and test-time optimizer fixed.

---

# T014 — Source-Supervised Sobolev Restoration Energy

**Status: OPEN.**

## Scientific question

Can explicit source-side supervision of the learned energy's **ISP-state gradient** repair the failure exposed by T013?

Test the causal hypothesis:

> T013 failed because scalar value regression underdetermined the local derivative field. If the same scalar energy is trained to match clean-reference restoration gradients on source states, then its label-free test-time gradient should become aligned enough to generate states that beat matched direct/discrete and frozen semantic fixed-depth controls.

This is a derivative-supervision experiment, not a feature expansion, architecture search, learned policy, detector objective, meta-initialization, or new spatial basis.

## Frozen components

Reuse unchanged:

- T006 frozen CLIP exposure readout and T007 frozen gate/calibration;
- T011 `region2` geometry, projected action box, inactive exact identity, global/bilinear controls, direct and projected-discrete baselines;
- dark EV `[0,+0.5]`, bright EV `[-0.5,0]`, gamma `[0.8,1.25]`;
- T012 frozen semantic `fixed_step_source=16` baseline;
- T013 **exact 28-feature schema** and `28 -> 64 -> 64 -> 1` SiLU energy architecture;
- T013 fixed source state-bank definition: identity, direct, discrete, semantic `{1,4,8,16,40}`, first 16 unscrambled 8-D Sobol states, duplicates retained, all-inactive identity only;
- learned-energy test-time optimization: identity reset, Adam `lr=0.03`, exactly 40 projected updates when active, save steps 0..40, choose minimum predicted energy with exact ties earliest;
- no WB/contrast and no change to synthetic degradation definitions.

Do **not** add image embeddings, step index, condition metadata, degradation masks/gains, image ID, clean statistics, new prompts, or additional energy features in T014. Holding the representation fixed is essential to isolate derivative supervision.

## Fresh source/development split

All 100 T013 train/calibration images are now permanent development data. Exclude every **508** image ID inspected through T013.

From official COCO val2017 image files only (no annotations), numeric ascending, original shorter side >=320, take the next eligible:

- first 80: `train_t014_sobolev`;
- next 20: `calibration_t014_sobolev`.

These 100 become permanent development data. Conditions stay `clean`, `homogeneous_dark`, `homogeneous_bright`, `left_right`, `quadrants`.

## Source derivative records

Build the same fixed T013 state bank on the 80 source-train images. For every bank state persist:

- the 28 raw test-time features `f`;
- `y = log(MSE_to_clean + 1e-6)`;
- the raw Region2 fast state `phi`;
- for active states, the exact source-only reference gradient
  `g_ref = ∇_phi log(MSE_to_clean + 1e-6)`;
- for active states, the exact feature Jacobian `J = ∂f/∂phi` in the same eight raw EV/gamma coordinates.

`g_ref` and `J` are **source-training supervision only**. They must never be serialized into a test-time energy input or API. Zero-norm reference-gradient rows do not contribute to the directional loss but still contribute to scalar value regression. All-inactive rows remain value-only identity rows.

An efficient cached-Jacobian implementation is allowed, but the cached `J` must numerically reproduce the autograd chain rule for a fixed fixture. No finite-difference target substitution and no outcome-dependent row sampling.

## Two fixed heads on the same source data

Train exactly two heads, both with the identical T013 architecture, normalization, initialization seed, optimizer, batch order, 100 epochs, final epoch only:

1. `value_only_control`: exact T013 standardized Huber value-regression objective;
2. `sobolev_primary`: the same value loss plus a fixed derivative-direction loss.

For `sobolev_primary`, compute the predicted raw-state gradient by the exact chain rule:

`g_E = J^T ∇_f E_psi(f)`.

Use

`L_dir = mean((1 - cosine(g_E, g_ref))/2)`

over active rows with nonzero `g_ref`, and

`L_total = L_value + L_dir`.

Weights are exactly `1 : 1`; no sweep, warm-up, curriculum, magnitude loss, or calibration-based reweighting. Keep T013 train-only feature/target normalization and AdamW `lr=1e-3`, weight decay `1e-4`, batch 256, seed 7, 100 epochs, final epoch only.

The value-only control and Sobolev head must be frozen before calibration inference. Report their train value loss and derivative cosine statistics separately; no model selection between them.

## Calibration methods

On the 20-image calibration split run, at minimum:

- `identity`;
- `region2_direct`;
- `region2_discrete_projected`;
- old `region2_ttt_projected` semantic final;
- frozen semantic `fixed_step_source=16`;
- `region2_ttt_energy_value_only` using the same-split value-only control;
- `global_ttt_energy_sobolev`;
- `bilinear2_ttt_energy_sobolev`;
- primary `region2_ttt_energy_sobolev`;
- reference-only `oracle_best_sobolev_checkpoint` after persistence.

Both learned energies use the same frozen 40-update/min-energy test-time procedure. The value-only head is a mechanism control, not a selectable fallback.

## Stage-A diagnostics and gate

At identity, for every active non-clean calibration episode, compute the same offline raw-coordinate reference-gradient cosine used in T013 for **both** learned heads. Persist the full distributions and per-condition counts.

Proceed to fresh evaluation only if **all eight** primary clauses hold for `region2_ttt_energy_sobolev`:

- positive reference-gradient cosine fraction `>= 0.80`;
- median reference-gradient cosine `>= 0.50`;
- clean p95 MSE `<= 0.005`;
- homogeneous dark MSE `<= 0.65 × identity`;
- homogeneous bright MSE `<= 0.65 × identity`;
- pooled heterogeneous MSE `<= 0.95 × region2_discrete_projected`;
- pooled heterogeneous MSE `<= 0.95 × frozen fixed_step_source=16`;
- pooled heterogeneous MSE `<= 0.95 × region2_ttt_energy_value_only`.

Additionally report, without making them alternate pass routes:

- Sobolev minus value-only positive-cosine fraction and median-cosine changes;
- full selected-step distributions;
- reference-only Sobolev trajectory oracle regret;
- oracle/discrete and oracle/fixed16 ratios;
- fraction of projected updates and final coordinates on action-box boundaries.

If any clause fails, **stop at Stage A**. Do not tune loss weights, Jacobian rows, bank, architecture, features, epochs, LR, action box, update budget, selector, or gates on these 20 images. Do not create/read a fresh T014 evaluation manifest.

## Freeze barrier and fresh Stage B

If all Stage-A clauses pass, first commit and verify an immutable Sobolev-energy receipt containing source manifests/hashes, bank definition, feature schema, Jacobian convention, derivative-loss equation, normalization, checkpoint/hash, training code SHA/recipe, Stage-A metrics, and frozen T006/T007/T011 identities. Verify the actual Git blob before loading any fresh image.

Only then create `evaluation_t014` from the next 40 eligible images after excluding all prior development/evaluation IDs; commit manifest/hash before outcomes. Run the five primary conditions plus `offset_left_right_40` as report-only stress.

Fresh qualification keeps the T013 restoration clauses and adds the same-split value-only control:

- clean mean `<=0.003`, clean p95 `<=0.005`;
- homogeneous dark/bright `<=0.60 × identity` each;
- pooled heterogeneous `<=0.85 × global_ttt_energy_sobolev`;
- pooled heterogeneous `<=0.95 × region2_direct`;
- pooled heterogeneous `<=0.95 × region2_discrete_projected`;
- pooled heterogeneous `<=0.95 × frozen fixed_step_source=16`;
- pooled heterogeneous `<=0.95 × region2_ttt_energy_value_only`;
- heterogeneous dark/bright regional MSE `<=1.05 × identity` each;
- exact-quadrant MSE `<=0.90 × bilinear2_ttt_energy_sobolev`.

`offset_left_right_40` stays report-only and cannot trigger a repair inside T014.

## Leakage/reproducibility requirements

At test time the Sobolev energy must receive **only** the same 28 T013 features. APIs must reject clean references, test/source labels, condition IDs, degradation masks/gains, annotations, image IDs, source Jacobians, and source reference gradients. Persist all label-free trajectories/features/energies/gradients/decisions/hashes before attaching clean-reference metrics/oracles. Add tests proving replacement reference/metadata cannot alter test-time trajectories and proving cached source `J` + head feature-gradient exactly reconstructs direct autograd `∇_phi E` on a fixture.

Run the full local suite and A6000 validation. If T014 qualifies, stop and report. If it fails, preserve the negative result without tuning the inspected split. Do not automatically start feature expansion, learned spatial basis, detector work, meta-initialization, prompt retraining, or ViT3.
