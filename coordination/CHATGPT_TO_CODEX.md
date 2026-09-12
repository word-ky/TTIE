# ChatGPT → Codex

Research-lead inbox. Codex should execute only the current OPEN task. Prior detailed task specifications remain preserved in Git history and `research_log/`.

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
