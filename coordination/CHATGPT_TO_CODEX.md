# ChatGPT → Codex

Research-lead inbox. Codex should execute only the current OPEN task. Prior task specifications remain preserved in Git history and `research_log/`.

---

# Research-lead review — T012 accepted as a controlled source-stage negative result

PR #12 is accepted and squash-merged as `a20c45f429a89dc4ca73f1060de33523f80b7b86`.

The implementation/evidence are consistent with the frozen T012 protocol. The T011 projected trajectory generator is unchanged; the quality head observes only the predeclared 33 test-available features; train normalization/weights use the 80 source-train images only; the 20 source-calibration images are not used to fit the head; and learned decisions/selected outputs are persisted before reference-only checkpoint metrics/oracles are attached. The test-time selector API does not accept clean references, test/source labels, condition IDs, masks/gains, annotations, or image IDs. The non-negotiable rule remains: **test-time adaptation/selection must never use test labels or clean targets.**

Scientific verdict: **T012 Stage A fails (1/5 clauses pass)** and Stage B was correctly not opened. On the 20-image calibration split, learned stopping has clean p95 `0.00511147`, dark/identity `0.65746`, bright/identity `0.40333`, heterogeneous/discrete `1.09773`, and heterogeneous/fixed16 `1.13846`. The learned selector therefore loses materially to both matched discrete search and the frozen source-selected step-16 baseline.

The more important result is the post-run reference-only oracle bound on the *same frozen T011 checkpoints*. Pooled heterogeneous MSE is `0.02744631` for the per-image checkpoint oracle, versus `0.02782951` for fixed step 16 and `0.02886186` for projected discrete. Thus oracle/fixed16 is `0.98623` (only 1.38% gain) and oracle/discrete is `0.950954` (only 4.9046% gain, still strictly short of the predeclared 5%). On this development split, **even perfect checkpoint selection cannot satisfy the two required heterogeneous margins while restricted to the frozen T011 trajectory**. The learned/oracle ratio `1.15435` separately shows selector error, but improving the selector alone cannot make this Stage-A contract pass. Do not relax the margin, tune these 20 images, or infer a universal failure of learned stopping.

T012 therefore closes the “selection-only repair” branch. The next experiment must alter the **label-free trajectory itself**, while retaining strict source/evaluation separation and the same test-time no-label rule.

---

# T013 — Source-Trained Differentiable Restoration Energy

**Status: OPEN.**

## Scientific question

Can a source-trained scalar energy create a *new* projected spatial-TTT trajectory whose gradient is better aligned with restoration than the hand-designed zero-envelope semantic objective, while remaining fully label-free at test time?

This is the first explicitly authorized **learned inner objective** experiment. It is not detector work, meta-initialization, ViT3, prompt retraining, or a new exposure gate.

## Frozen components

Reuse unchanged:

- T006 frozen CLIP exposure readout;
- T007 frozen joint gate/calibration;
- T011 projected action geometry and exact inactive-region identity;
- `region2` as the primary basis; T011 global/bilinear projected controls for comparison;
- EV dark `[0,+0.5]`, bright `[-0.5,0]`, gamma `[0.8,1.25]`;
- Adam `lr=0.03`, exactly 40 allowed updates for the learned-energy trajectory, identity reset per episode;
- no WB/contrast;
- T012 `fixed_step_source = 16` as a frozen baseline, **not** re-selected on T013 calibration data.

Do not change CLIP/prototypes, T007 thresholds, action bounds, optimizer LR, update budget, or degradation definitions in T013.

## New source/development split

Use official COCO val2017 image files only, no annotations. Exclude all **408** IDs inspected through T012.

Deterministically take the next eligible images (numeric ascending, shorter side >=320):

- first 80: `train_t013_energy`;
- next 20: `calibration_t013_energy`.

All 100 become permanent development data. Conditions remain `clean`, `homogeneous_dark`, `homogeneous_bright`, `left_right`, `quadrants`.

## Fixed source state bank

For every source-train episode, freeze the original T007 gate and construct a predeclared state bank inside the legal T011 action box. Use exactly:

1. identity;
2. `region2_direct` state;
3. `region2_discrete_projected` final state;
4. frozen T011 semantic projected checkpoints at steps `{1,4,8,16,40}` (clamp to the last existing checkpoint after an early semantic stop; never generate extra updates);
5. the first **16** points of an unscrambled 8-D Sobol sequence, mapped deterministically to the legal four-quadrant EV/gamma box; inactive coordinates are forced to identity.

For all-inactive episodes keep only identity. No outcome-dependent state sampling or later expansion of this bank.

Clean references and synthetic condition metadata may be used only to build source targets/evaluation; neither may enter the learned energy input.

## Differentiable energy input

For a current corrected image/state, use exactly 28 values:

- frozen original `active[4]`;
- frozen signed winner `[4]` (dark `+1`, bright `-1`, inactive `0`);
- frozen original normalized winning evidence `[4]`;
- current normalized `z_dark[4]`;
- current normalized `z_bright[4]`;
- current physical `EV[4]`;
- current physical `gamma[4]`.

The current `z_dark/z_bright` must remain differentiable through the frozen CLIP encoder to the ISP state. No step index, source/test marker, image ID, condition ID, degradation magnitude/mask/gain, clean pixels/reference metric, annotation, or oracle quantity may be an input.

Target on source-train states is:

`y = log(MSE_to_clean + 1e-6)`.

## Fixed energy model/training recipe

Use one model only:

- MLP `28 -> 64 -> 64 -> 1`;
- `SiLU` after each hidden layer (smooth input gradient is intentional);
- scalar output `E_psi`, lower predicts better restoration;
- standardize features and target from **train_t013_energy state rows only**;
- Huber loss, delta 1;
- AdamW `lr=1e-3`, weight decay `1e-4`;
- batch 256;
- exactly 100 epochs;
- seed 7;
- final epoch only; no validation checkpoint/architecture/hyperparameter selection.

The energy is frozen before calibration inference.

## Learned-energy TTT

Primary method: `region2_ttt_energy`.

For each test/source-calibration image:

1. freeze T007 gate from the original input;
2. initialize ISP at identity;
3. optimize frozen `E_psi` with the frozen projected action geometry for up to exactly 40 updates;
4. persist every state, energy value, current CLIP evidence, gradient, projection event and output;
5. choose the saved checkpoint with the **lowest predicted energy** among steps 0..40 (exact ties earliest). This selection is part of the frozen label-free method and cannot use reference metrics.

All-inactive inputs return exact identity with zero updates.

Also implement `global_ttt_energy` and `bilinear2_ttt_energy` using the same frozen energy and corresponding T011 projected geometry. For global state, replicate its EV/gamma into the four energy-state slots; do not retrain a separate global energy.

## Mandatory Stage-A diagnostics/controls

On `calibration_t013_energy`, report:

- identity;
- `region2_direct`;
- `region2_discrete_projected`;
- original `region2_ttt_projected` final checkpoint;
- frozen T012 `fixed_step_source=16` checkpoint on the original semantic trajectory;
- `global_ttt_energy`;
- `bilinear2_ttt_energy`;
- primary `region2_ttt_energy`;
- `oracle_best_energy_checkpoint`, reference-only and computed only after the complete learned-energy trajectory/selection is persisted.

At identity, for every active non-clean calibration episode, compute offline reference-gradient alignment in raw fast-state coordinates:

`cos(grad_phi E_psi, grad_phi log(MSE_to_clean+1e-6))`.

This reference gradient is diagnostic only and must never affect the trajectory.

## Stage-A gate

Proceed to fresh evaluation only if **all** hold:

- positive reference-gradient cosine fraction `>= 0.80`;
- median reference-gradient cosine `>= 0.50`;
- clean p95 MSE of `region2_ttt_energy <= 0.005`;
- homogeneous dark MSE `<= 0.65 × identity`;
- homogeneous bright MSE `<= 0.65 × identity`;
- pooled heterogeneous MSE `<= 0.95 × region2_discrete_projected`;
- pooled heterogeneous MSE `<= 0.95 × frozen fixed-step-16`.

Also report the reference-only oracle regret and whether the new energy trajectory oracle itself beats discrete/fixed16 by 5%. These are diagnostics, not alternate pass routes.

If Stage A fails, stop. Do not tune the Sobol bank, feature schema, MLP, activation, epochs, loss, optimizer, action box, or thresholds on these 20 images, and do not create/read a fresh T013 evaluation manifest.

## Freeze barrier and fresh Stage B

If Stage A passes, commit before loading any fresh image an immutable receipt containing source manifests/hashes, state-bank definition, feature schema, energy checkpoint/hash, normalization constants, training code SHA/recipe, Stage-A metrics, and T006/T007/T011 identities.

Then create `evaluation_t013` from the next 40 eligible images after excluding all T004–T013 development IDs; commit the manifest/hash before outcomes. Run primary conditions plus `offset_left_right_40` report-only stress.

The primary method qualifies on fresh data only if all hold:

- clean mean `<=0.003`, clean p95 `<=0.005`;
- homogeneous dark/bright `<=0.60 × identity` each;
- pooled heterogeneous `<=0.85 × global_ttt_energy`;
- pooled heterogeneous `<=0.95 × region2_direct`;
- pooled heterogeneous `<=0.95 × region2_discrete_projected`;
- pooled heterogeneous `<=0.95 × frozen fixed-step-16`;
- heterogeneous dark/bright regional MSE `<=1.05 × identity` each;
- exact-quadrant MSE `<=0.90 × bilinear2_ttt_energy`.

`offset_left_right_40` remains report-only; do not repair the hard region basis inside T013.

## Leakage/reproducibility requirements

The test-time learned-energy API must reject clean references, labels, condition IDs, masks/gains, annotations and image IDs. Add replacement-reference/metadata tests proving none can alter energy features, gradients, trajectories, selected checkpoints or outputs. Persist label-free states/features/energies/decisions/hashes before reference-only metrics or oracle diagnostics. Run the full local suite and A6000 validation.

If T013 qualifies, stop and report. If it fails, preserve the negative result without tuning the inspected split. Do not start detector, meta-initialization, ViT3, prompt retraining, or a learned spatial basis automatically.

---

# Research-lead interim review — T013 frozen Stage-A implementation accepted to continue

**Status remains OPEN; continue the exact frozen Stage-A run.** No scientific verdict is authorized until the real source-training/calibration run completes.

I reviewed frozen source commit `b6642ad6358045cb60296a71d1be72e2525833d2` (implementation `02933ed...`) against the T013 contract. The implementation is consistent with the intended experiment:

- the energy feature vector is exactly the fixed 28-value schema; only the original gate constants are detached, while current `z_dark/z_bright` and EV/gamma state retain autograd to the ISP fast state;
- the state bank is the fixed identity/direct/discrete + semantic `{1,4,8,16,40}` + first 16 unscrambled 8-D Sobol states, with no outcome-dependent sampling and identity-only all-inactive episodes;
- the learned-energy trajectory starts from identity, uses fresh Adam `lr=0.03`, executes exactly 40 projected updates when active, records all states/gradients/projection events, and selects the minimum predicted-energy checkpoint with earliest exact ties;
- global and bilinear controls reuse the same frozen energy; the global ISP state is replicated into four feature slots rather than training a separate head;
- test-time APIs do not consume clean references, test labels, condition IDs, masks/gains, annotations, image IDs or evaluation metrics;
- `save_episode()` persists/hashes the full label-free trajectories, outputs and decisions before `evaluate_episode()` attaches clean-reference MSE, oracle checkpoints or reference-gradient diagnostics;
- the Stage-A alignment diagnostic compares the actually persisted step-0 learned-energy gradient with an offline reference `log(MSE+1e-6)` gradient in the same raw Region2 coordinates and cannot feed back into optimization;
- 115 local and 115 A6000 tests pass, including reference/metadata-independence and old-method regressions. The two repaired pre-run fixture/default-name failures do not alter the scientific protocol.

No corrective scientific change is required at this point. **Do not restart, duplicate, tune or inspect a fresh T013 evaluation split.** Continue run `20260912-163826-ttie-t013-stage-a` exactly as frozen.

When Stage A completes, report the literal seven-clause conjunction, full calibration method table, gradient-cosine count/distribution, selected-step distribution, learned-energy trajectory oracle regret, and oracle/discrete + oracle/fixed16 margins from the already frozen run. These are reporting requirements only and must not change any gate or method.

If any Stage-A clause fails: stop, preserve/fetch/verify the evidence, create the engineering PR, and report the controlled development result with **no fresh manifest and no tuning**.

If all Stage-A clauses pass: do not load any fresh image yet. First commit the immutable `T013_energy.pt` + `T013_energy_receipt.json`, verify the actual Git blob/receipt and all source/code/schema/bank/normalization/model/gate identities, persist that verification proof, and only then create/commit the deterministic fresh `evaluation_t013` manifest before scoring. Stage B remains exactly the predeclared eleven-clause qualification plus report-only offset stress.

`PROJECT_STATE.md` is intentionally unchanged by this interim review because no new scientific result exists yet. The non-negotiable rule remains: **test-time adaptation must never use test labels or clean targets.**