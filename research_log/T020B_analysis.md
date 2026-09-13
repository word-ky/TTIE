# T020-B development target positive

All five predeclared clauses pass on the fixed40 development images /120 clean, homogeneous-dark and homogeneous-bright episodes. Outcomes are **61 beneficial /59 equal /0 harmful**. This is an explicitly reference-only development target audit, not fresh qualification, not learned prediction and not a deployable label-free selector. T020-A remains a valid fresh safety negative; no selector was trained or changed.

## Results

| Group | n | H0 | H_delta | H* | H_delta/H0 | H*/H0 | Beneficial/equal/harmful |
|---|---:|---:|---:|---:|---:|---:|---|
| nonspatial_pool | 120 | 0.021385082408475378 | 0.019265628350876796 | 0.019217849626572085 | 0.9008910034988411 | 0.89865679540031274 | 61/59/0 |
| clean | 40 | 0.00028852283139713109 | 0.00020819390465476317 | 0.00020819390465476317 | 0.72158554540246811 | 0.72158554540246811 | 2/38/0 |
| homogeneous_dark | 40 | 0.045241762069053948 | 0.041020679031498732 | 0.040937563660554586 | 0.9066994112406046 | 0.90486227300497879 | 26/14/0 |
| homogeneous_bright | 40 | 0.018624962324975059 | 0.016568012116476894 | 0.016507791314506904 | 0.889559496947819 | 0.88632615875796195 | 33/7/0 |

Literal acceptance vector `[true,true,true,true,true]`: pooled and each of the three condition means satisfy `H_delta<=1.01H0`, and total harmful count is0. No rounding relaxation. All harmful-case and harmful combined-movement lists are empty. Oracle statistics are descriptive and do not enter any acceptance condition or target axis choice.

| Group | No move | x-only | y-only | Both | Center in exact oracle tie set |
|---|---:|---:|---:|---:|---|
| nonspatial_pool | 59 | 7 | 10 | 44 | 57/120 (0.47499999999999998) |
| clean | 38 | 0 | 0 | 2 | 38/40 (0.94999999999999996) |
| homogeneous_dark | 14 | 1 | 5 | 20 | 14/40 (0.34999999999999998) |
| homogeneous_bright | 7 | 6 | 5 | 22 | 5/40 (0.125) |

| Group | x center/lower/upper | y center/lower/upper |
|---|---|---|
| nonspatial_pool | 69/30/21 | 66/35/19 |
| clean | 38/0/2 | 38/0/2 |
| homogeneous_dark | 19/14/7 | 15/17/8 |
| homogeneous_bright | 12/16/12 | 13/18/9 |

The target improves pooled mean MSE by9.9109% relative to the cached canonical center. Clean has38 no-move/equal cases and two beneficial both-axis targets. The non-spatial target is safe on this development set, but non-spatial does not imply literally zero geometric headroom: homogeneous-dark and homogeneous-bright nine-hard oracle ratios are about0.9049 and0.8863. This describes correction of the fixed cached Region2 state, not an unconditional property of the original image or a deployable detector of spatiality.

## Exact rule and data scope

Use the accepted T019-A `delta=.01` rule independently for each axis, with reference MSE only. The five cross positions are center(.5,.5), x-lower(.4,.5), x-upper(.6,.5), y-lower(.5,.4), y-upper(.5,.6). The nine hard candidates are bx-major/by-minor over(.4,.5,.6), all tau0. If the best relative improvement on an axis is below1%, stay center; otherwise select minimum MSE with center→lower→upper tie order. A qualifying positive gain makes center strictly worse, so the original implementation's lower-before-upper branch is exactly equivalent. After independent axis choices, look up that combined hard state with no joint reoptimization.

For H0=0, both axes stay center, even when alternatives are also zero; gains are recorded as null. There is no epsilon or tuned denominator. This required zero branch is the only rule extension beyond T019-A's positive-H0 implementation; the positive branch is identical and tested against the original function extracted from accepted PR32 head91f750e871d2c133624bbe4972f3fb3086f25a6c. This audit's per-condition zero-H0 counts are:

- clean: 38.
- homogeneous_dark: 0.
- homogeneous_bright: 0.

No new/fresh images are selected. The40 IDs are the exact accepted T018/T019 geometry development set (55167 through60886 in the fixed manifest), originally cached in T015 run20260912-213014-ttie-t015-fresh-ready. `T020B_source_inputs/origins.json` binds six original Git commit:path inputs: development manifest; cache config/manifest; accepted T016 config; accepted T018 development candidate table; and original T019-A rule source. The T016 receipt pins the exact T015 manifest SHA77f405417324fc4e2d50ccd4606caa06725416c9b0f5182daac5dbdc067a4a09, cache-manifest SHAb3c132837872ba392b23caee5ebb81abbdfe5a67aa770689b243ec775231905e and cache-config SHAabff3db7ab9189c8288271f2103bc78bdfb46fa3890c15a6e8a472c602bfa91e. The original T018 development table proves the same40 IDs; the full ordered list is in the lock and independent receipt.

Conditions are the accepted T014 clean/dark/bright gains1/.45/1.55 and existing clipping/preprocessing, in that fixed order. T006/T007 numeric gate parameters, T014 energy SHA c3d1eef9f20af163e1268823cf16d3fdaed315e7723fc94db3b1138ad1336521, canonical identity-start Region2 trajectory,40 projected label-free updates when active and checkpoint rule are inherited unchanged from the byte-pinned cache. All68 baseline scientific files are unchanged. No trajectory, CLIP feature, model, normalizer or selector is retrained/recomputed.

## Execution, information boundary and verification

The run reads only original development inputs and cache files. Each of120 cached identity inputs, selected Region2 outputs, trajectories and decisions is checked against the accepted artifact manifest. The selected grid must equal the cached trajectory grid at `selection.selected_step`. Every canonical hard render reproduces the old selected output **pixel-for-pixel**, and its MSE equals the cached canonical output's directly evaluated MSE exactly. GPU1 renders the same nine hard candidates from those frozen corners; it never optimizes them. Source/reference pixels are used only for this explicitly authorized development diagnostic. No T020-A fresh references, per-row features/logits/outcomes, harmful-row-specific rule, new subset or confidence/spatiality gate is used. Only the accepted aggregate conclusion of broad safety failure motivated this fixed task.

Final scientific source `7134383cffd3abd8fe71090b3498bba6f837cb94`; candidate table SHA `0c8c9c0ae20f590db1ecfb39f82540824e0a6c52ce43b0ff09e1231ac4fafc79`, cache bindings SHA `17bdf8e38105286960247deeef89c4cd91cddd266d4947e984ccfb897d9a345f`. Table frozen `2026-09-13T11:32:17.385010+00:00` before fixed target reporting at `2026-09-13T11:32:17.391800+00:00`. Independent verification passed `2026-09-13T11:33:26.558977+00:00`: all120 rows; exact positive/zero 1% rule; tie order; x/y labels; combined lookup; all five booleans; movement/outcome/label counts; exact oracle ties/ratios; six original Git origins;78 runtime/source files; all table/config/cache/state bindings. The verifier imports no TTIE or model library and performs no new pixel experiment.

Completed GPU run20260913-193203-ttie-t020b-target-audit-fixed in release20260913-193127-ttie-t020b-ready exits0. GPU1 is NVIDIA RTX A6000 / Torch2.4+cu121; CPU performs small metadata/reference arithmetic only. The original480 used cached files total1225337177bytes and remain in the existing project run storage;120 compact corner states and all new tables/receipts/logs are retained.

Baseline11 tests passed14.00s. New fixed-rule tests3 passed3.45s; affected11 tests passed22.86s. An observed cache-reader bug caused the first run20260913-192817-ttie-t020b-target-audit to exit1 at its first row before any candidate/target result: original `selected_step` is nested under `selection`. The one-line reader fix is source7134383c (initialsource7a10d518), then3 focused tests passed15.51s and the complete120-case integration run passed. No data, threshold, candidate, target or scientific criterion changed. Both failed and completed runs are preserved. Separately, an intermediate deployment preceded completion of its staging copy and lacked Git pack/index; no job ran there. Redeploying the completed stage produced the verified78-file release. A preliminary whole-gate-dict comparison also differed only because the original receipt contains additional provenance fields; numeric q_joint/tau/scale were exactly equal. These were engineering corrections, not scientific retuning.

Commands: `python -m pytest tests/test_nonspatial_target.py tests/test_nonspatial_safety.py -q`; repair-focused `python -m pytest tests/test_nonspatial_target.py -q`; `bash scripts/run_t020b_a6000.sh 7134383cffd3abd8fe71090b3498bba6f837cb94`; independent `python research_log/T020B_verify.py --output research_log/remote_runs/20260913-193203-ttie-t020b-target-audit-fixed/artifacts/audit --receipt research_log/T020B_verification.json`. Exact remote command is also in run meta.json.

Interpretation: the fixed target passes the stated safety/zero-harm requirement on non-spatial development. This supports the target-safe explanation on these development cases and leaves learned movement generalization/calibration as an unresolved issue; it does not prove causality for the fresh failure or guarantee target safety on unseen images. T014 remains the broad deployable baseline. Stop for research-lead review; do not train/modify T019-C or start T020-C, a gate, a new cohort or downstream benchmark in this cycle.
