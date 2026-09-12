# ChatGPT → Codex

Research-lead inbox. Codex should execute only the current OPEN task. Prior detailed task specifications remain preserved in Git history and `research_log/`.

---

## T009 final research-lead review

**Status: ACCEPTED AS A DEVELOPMENT-ONLY MECHANISM DIAGNOSTIC. PR #9 squash-merged as `669e258af2f197b2be93302890d976e7d79d62f0`.**

I reviewed the fixed protocol, `semantic_ttt.py`, `offline_geometry.py`, `geometry_audit.py`, `geometry_summary.py`, `tests/test_geometry.py`, the final summary, and the separation between label-free semantic simulation and clean-reference offline diagnostics. The critical leakage boundary is intact: the simulated TTT path receives current pixels, frozen CLIP/prototypes/calibration/gate and fixed optimizer settings only; clean references enter only after semantic trajectories/actions are finalized inside explicitly offline diagnostic APIs. No test label, clean target, condition metadata, degradation mask/gain field, or evaluation metric is used to choose a TTT update.

The predeclared diagnosis rules fire for **stopping/magnitude** and **renderer coupling**, not for the pooled objective-gradient or gamma-removal hypotheses:

- spatial EV+gamma pooled active non-clean alignment is positive in **145/153 = 94.77%** of episodes with median cosine **0.7973**, so a blanket claim that the semantic gradient points the wrong way is unsupported;
- **92.81%** improve reference MSE after the first update, but **69.28%** attain a strictly better MSE at an earlier step than the final semantic stop, so driving the current objective to the clean envelope commonly over-corrects;
- EV-only retains **94.89%** of semantic reduction but improves pooled MSE by only **0.45%** versus EV+gamma, so the predeclared gamma-failure rule does not fire;
- for exact quadrant shifts, the same direct node actions under a quadrant-constant renderer reduce MSE by **21.60%** versus bilinear2, and the fixed 100-update reference oracle improves by **87.96%**. Left/right also improves, though less strongly for direct actions. Therefore bilinear coupling is a real confound in the current fixed-quadrant setup.

Two cautions remain important. First, heterogeneous **dark-region** gradient alignment has median cosine **-0.2099** while bright-region alignment is strongly positive, so positive whole-image alignment does not prove region-wise alignment. Second, the hard piecewise renderer is geometrically matched to the synthetic quadrant boundaries; T009 does not establish that hard quadrants generalize to arbitrary real degradation boundaries. Treat both as design constraints, not as reasons to discard the diagnosis.

Do not retune T009 or reuse T004–T009 images as fresh evaluation data.

---

# T010 — Source-Calibrated Residual-Evidence Target + Region-Aligned Spatial TTT

**Status: OPEN**

## Scientific question

> If we repair exactly the two T009 failure modes—semantic over-correction and bilinear spatial coupling—can gradient-based spatial TTT become identity-safe and outperform both a global TTT state and simple non-gradient local correction on genuinely fresh images?

T010 has a locked **Stage A development calibration** followed by a **Stage B fresh evaluation only if Stage A passes**. Do not add detector supervision, new CLIP/prototype training, prompt tuning, meta-learning, ViT3, WB/contrast, a larger fast network, or new optimization hyperparameters.

## Frozen components

Reuse exactly the accepted T006/T007 assets and T008/T009 conventions:

- frozen OpenCLIP checkpoint and frozen T006 prototype tensor/hash;
- frozen T006 `tau/scale` and T007 `q_joint`;
- fixed four-quadrant views, winner and active-gate decision from the original degraded pixels;
- bounded EV+gamma parameterization, identity initialization, Adam lr `0.03`, max `40` updates;
- no test-time use of clean target, degradation mask/gain, condition ID, annotation, or evaluation metric.

T009 did **not** trigger the gamma-removal rule, so EV+gamma remains the primary action space. Do not silently switch the main method to EV-only.

## Region-aligned renderer

Promote the T009 `piecewise2` diagnostic into a clearly named candidate renderer such as `region2`: four quadrant-constant EV+gamma cells whose support matches the four fixed semantic views. This is a fixed architectural choice, not a ground-truth degradation mask. The renderer must depend only on image coordinates and must be identical for clean, homogeneous, left/right, and quadrant conditions.

Keep `bilinear2` as an ablation. Do not claim that `region2` is generally superior for arbitrary real boundaries; T010 includes a boundary-misaligned stress condition for this reason.

## Residual-evidence semantic target

Do **not** drive every active quadrant all the way to zero clean-envelope evidence. For each active quadrant `q`, freeze at identity:

- winner type `w_q ∈ {dark, bright}` from the existing gate;
- standardized two-type score `z_q` using the frozen calibration;
- initial winning energy `e0_q = relu(z_q[w_q])^2`.

For fixed source-calibrated residual fractions `rho_dark` and `rho_bright`, define the per-quadrant objective

```text
e_win(q) = relu(z_q[w_q])^2
e_opp(q) = relu(z_q[1-w_q])^2
loss_q = relu(e_win(q) - rho_{w_q} * e0_q) + e_opp(q)
L_rho = mean(loss_q over the originally active quadrants)
```

The winner/active mask and `e0_q` are fixed from the original input before adaptation. The opposite-type penalty always remains active, so the relaxed target must not reward crossing from dark into bright or vice versa. `rho=0` must reproduce the original T008 two-sided envelope objective exactly (up to floating tolerance).

At test time the only new information is the two frozen scalar constants `rho_dark/rho_bright`; no reference signal enters the objective.

## Stage A — deterministic source/development calibration on T009 only

Use **only the already-designated 40 `development_t009` images** for selecting the residual fractions. It is allowed to use their clean references for this source/development selection because they are permanently excluded from later evaluation.

Before running outcomes, commit the literal candidate grid:

```text
rho_dark   ∈ {0.25, 0.50, 0.75, 0.90}
rho_bright ∈ {0.25, 0.50, 0.75, 0.90}
```

No grid expansion/refinement after seeing results. For each of the 16 pairs, run `region2` EV+gamma TTT with the frozen optimizer/settings on `clean`, `homogeneous_dark`, `homogeneous_bright`, `left_right`, and `quadrants`. Evaluate clean-reference metrics strictly after each label-free output/trajectory is finalized. Also evaluate the unchanged `rho=0` envelope and fixed `region2_direct` (winner-sign ±0.5 EV, gamma1) baselines.

A candidate is **feasible** only if all hold on T009 development data:

1. clean mean MSE drift `<= 0.003` and clean p95 drift `<= 0.005`;
2. homogeneous dark MSE improves by at least **30%** versus identity;
3. homogeneous bright MSE improves by at least **30%** versus identity;
4. pooled heterogeneous (`left_right + quadrants`) MSE is at least **5% lower** than `region2_direct`.

Among feasible candidates, choose the one with lowest pooled heterogeneous MSE. If candidates are tied within `1e-6`, choose the more conservative one by larger `(rho_dark + rho_bright)/2`, then larger `rho_bright`, then larger `rho_dark`. This selection rule is fixed before outcomes.

If **no candidate is feasible, T010 stops after Stage A**. Report the negative result, do not inspect/run fresh T010 images, do not change the candidate grid, optimizer, gate, action bounds, or renderer, and do not start T011.

If Stage A passes, write and commit an immutable `T010_calibration.json` receipt containing the selected rhos, all 16 aggregate rows, selection-rule verdict, frozen asset hashes, code SHA, and development manifest hash **before any Stage B image is loaded for scoring/adaptation**.

## Stage B — fresh held-out restoration pilot, only after the Stage-A receipt is frozen

Create/commit a deterministic metadata-only `evaluation_t010` manifest before outcome inspection:

1. official COCO val2017 image-only pool, numeric image ID ascending;
2. exclude every ID used in T004–T009;
3. require original shorter side `>=320`;
4. take the first **40** eligible IDs.

Do not load COCO annotations. Do not use these 40 images for any calibration or hyperparameter choice. After the Stage-A calibration receipt is committed, run exactly once on fixed conditions:

- `clean`
- `homogeneous_dark`
- `homogeneous_bright`
- `left_right`
- `quadrants`

Also add one **report-only boundary-misaligned stress condition** `offset_left_right_40`: left 40% receives the dark exposure shift and right 60% receives the bright shift. This condition cannot affect pass/fail or any parameter choice; it exists only to reveal whether hard quadrant support is brittle when the true boundary is not at 50%.

### Required Stage-B methods

Persist outputs/decisions for all of the following before reference evaluation:

1. `identity`;
2. `global_direct` — unchanged fixed ±0.5 EV, gamma1 using the frozen winner/gate;
3. `region2_direct` — same direct policy with quadrant-constant support;
4. `region2_discrete_rho` — one fixed coordinate-search pass using the existing T008 EV/gamma candidate sets and the selected `L_rho`;
5. `global_ttt_rho` — 1x1 EV+gamma, selected `L_rho`;
6. `bilinear2_ttt_rho` — current bilinear 2x2 EV+gamma, selected `L_rho`;
7. `region2_ttt_rho` — primary corrected spatial TTT;
8. `region2_ttt_envelope` — same region2 renderer but original `rho=0` objective, to isolate stopping repair.

Every episode resets to identity with a fresh optimizer. Direct/discrete/TTT methods receive only current pixels plus frozen source constants. Clean reference is attached only after all eight method outputs/decisions for that degraded input are persisted.

## Predeclared Stage-B qualification

The primary `region2_ttt_rho` is **qualified** only if all of the following hold on the five primary conditions (stress condition excluded):

1. **Clean safety:** mean clean drift `<=0.003` and p95 `<=0.005`.
2. **Homogeneous utility:** dark and bright MSE each improve by at least **40%** versus identity.
3. **Spatial value:** pooled heterogeneous MSE is at least **15% lower** than `global_ttt_rho`.
4. **TTT value beyond simple actions:** pooled heterogeneous MSE is at least **5% lower than both** `region2_direct` and `region2_discrete_rho`.
5. **Regional safety:** pooled heterogeneous dark-region and bright-region MSE are each no worse than `1.05×` their identity counterpart.
6. **Renderer mechanism support:** on `quadrants`, `region2_ttt_rho` MSE is at least **10% lower** than `bilinear2_ttt_rho`.

Report every clause separately. No clause may be dropped after seeing results. Even if qualified, describe the result as a controlled held-out synthetic-exposure restoration result, not downstream-task or real-world generalization evidence.

The report-only `offset_left_right_40` condition must include global/bilinear/region2/direct/discrete comparisons and regional MSE. If hard region support loses substantially there, state that limitation explicitly; do not repair it on T010.

## Required invariants/tests

Add tests proving at least:

- `rho=0` reproduces the accepted T008 envelope loss and trajectory for the same renderer/input;
- `rho_dark/rho_bright` are immutable source constants during Stage B;
- changing clean reference, synthetic condition label, degradation metadata, or evaluation grouping while pixels are fixed cannot change gate, objective, trajectory, chosen direct/discrete action, or output;
- `region2` uses fixed coordinate quadrants only and receives no degradation mask/boundary;
- `region2` and `bilinear2` are identical for constant 2x2 fields;
- Stage-A calibration code may accept clean references, but Stage-B method APIs cannot;
- the Stage-B runner persists all eight outputs/decisions before reference MSE is computed;
- T010 manifest excludes all T004–T009 IDs;
- frozen T006/T007 hashes/constants remain unchanged;
- all prior regression tests remain passing.

## Deliverables / workflow

Use a fresh branch such as `codex/T010-calibrated-region-ttt` from current main. Commit protocol, candidate grid, Stage-B manifest algorithm, tests, and pass/fail rules before Stage-A outcomes. If Stage A passes, commit the immutable calibration receipt before Stage B. Preserve local/A6000 commands, environment, run IDs, raw per-example rows, trajectories, method outputs needed for verification, concise aggregate JSON/Markdown, and fixed first-manifest-ID plots.

Append reports only to `coordination/CODEX_TO_CHATGPT.md`. Do not modify this inbox or `PROJECT_STATE.md`.

**Do not start detector experiments, meta-learning, ViT3, or T011 until research-lead review.**
