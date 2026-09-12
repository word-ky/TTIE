# T016-A: fixed-action renderer-transfer screen

Result: **strong_adaptive_basis_evidence**. Development-only, 120 existing episodes and exactly 27 renderers.

The oracle lowers aggregate MSE by 7.24% versus Region2 and 5.40% versus the accepted T015 three-basis oracle. This is reference-only headroom on inspected development data, not a deployable label-free selection result. The best single renderer remains the original hard Region2. Oracle selections use tau=0 in 113/120 cases (including 41 canonical hard and 72 shifted hard), tau=.05 in 4 and tau=.10 in 3. This supports investigating boundary placement; it does not establish a benefit from sigmoid smoothing itself. Twelve episodes have tied oracle minima, so selection counts are not unique preferences. Sixty-nine episodes have positive gain; the other 51 have zero gain. Offset cases provide the largest aggregate improvement (15.52% versus Region2), while exact quadrants barely change and remain 1.30% worse than the T015 three-basis oracle.

Code commit: `05e7dcf268c7b0479e6edabdc6ec76528d5730bf`. No CLIP, energy, checkpoint selection, optimizer, projection or action fitting was rerun.

This measures transfer of saved Region2 corner actions, not the capacity of an independently optimized soft basis.

## Sanity checks

`{'inputs': 120, 'max_pixel_abs': 0.0, 'max_mse_diff': 0.0, 'identity_checks': 3240, 'saved_corner_checks': 120, 'passed': True}`

Every renderer receives the identical saved physical corner grid. Exact identity is preserved. All nesting checks passed before candidate headroom was computed.

Global best fixed index: 12; (bx, by, tau) = (0.5, 0.5, 0.0). This same global choice is used in all per-condition ratios.

| Group | Region2 | Best fixed soft | Oracle soft | T015 oracle | Fixed/Region2 | Oracle/Region2 | Oracle/fixed | Oracle/T015 oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| spatial_pool | 0.0350435737482 | 0.0350435737482 | 0.0325076995068 | 0.0343645194507 | 1 | 0.927636540167 | 0.927636540167 | 0.945966945748 |
| left_right | 0.0333970155101 | 0.0333970155101 | 0.0321141692344 | 0.0330589110497 | 1 | 0.961587996528 | 0.961587996528 | 0.97142247626 |
| quadrants | 0.0309838496498 | 0.0309838496498 | 0.0309821883566 | 0.0305849858909 | 1 | 0.999946381963 | 0.999946381963 | 1.01298684482 |
| offset_left_right_40 | 0.0407498560846 | 0.0407498560846 | 0.0344267409295 | 0.0394496614113 | 1 | 0.844830981931 | 0.844830981931 | 0.872675194104 |

## All fixed renderers and oracle selection counts

Candidate and exact-MSE tie order is bx ascending, then by ascending, then tau ascending. All candidates, including hard/shifted candidates, remain in the predeclared family.

| Index | bx/by/tau | Spatial MSE | LR MSE | Quadrants MSE | Offset MSE | Oracle counts (all/LR/Q/offset) |
|---|---|---:|---:|---:|---:|---|
| 0 | (0.4, 0.4, 0.0) | 0.0404904491035 | 0.0403534304351 | 0.0453183843754 | 0.0357995325001 | 25 / 2 / 0 / 23 |
| 1 | (0.4, 0.4, 0.05) | 0.0407923528692 | 0.040025038342 | 0.044301971863 | 0.0380500484025 | 1 / 0 / 0 / 1 |
| 2 | (0.4, 0.4, 0.1) | 0.0431085930672 | 0.0417378788814 | 0.0473218530416 | 0.0402660472784 | 0 / 0 / 0 / 0 |
| 3 | (0.4, 0.5, 0.0) | 0.0386046610928 | 0.0406808688072 | 0.03894816963 | 0.0361849448411 | 2 / 0 / 0 / 2 |
| 4 | (0.4, 0.5, 0.05) | 0.0400544804909 | 0.0403627898777 | 0.0414266664302 | 0.0383739851648 | 0 / 0 / 0 / 0 |
| 5 | (0.4, 0.5, 0.1) | 0.0428169388091 | 0.0420508524869 | 0.0458744586213 | 0.0405255053192 | 0 / 0 / 0 / 0 |
| 6 | (0.4, 0.6, 0.0) | 0.0406867082696 | 0.0409095321316 | 0.0447310128715 | 0.0364195798058 | 11 / 0 / 0 / 11 |
| 7 | (0.4, 0.6, 0.05) | 0.0410311906831 | 0.0406678748783 | 0.0438224578509 | 0.0386032393202 | 0 / 0 / 0 / 0 |
| 8 | (0.4, 0.6, 0.1) | 0.0433550716611 | 0.0423564487137 | 0.0469621150754 | 0.0407466511941 | 0 / 0 / 0 / 0 |
| 9 | (0.5, 0.4, 0.0) | 0.0375268205302 | 0.0330234457506 | 0.0391216061078 | 0.0404354097322 | 20 / 20 / 0 / 0 |
| 10 | (0.5, 0.4, 0.05) | 0.0390991452693 | 0.0361478280742 | 0.0415566777112 | 0.0395929300226 | 1 / 1 / 0 / 0 |
| 11 | (0.5, 0.4, 0.1) | 0.0420200443283 | 0.0393177286023 | 0.0459145188332 | 0.0408278855495 | 0 / 0 / 0 / 0 |
| 12 | (0.5, 0.5, 0.0) | 0.0350435737482 | 0.0333970155101 | 0.0309838496498 | 0.0407498560846 | 41 / 2 / 39 / 0 |
| 13 | (0.5, 0.5, 0.05) | 0.0381645783239 | 0.0364768592641 | 0.0381196441362 | 0.0398972315714 | 0 / 0 / 0 / 0 |
| 14 | (0.5, 0.5, 0.1) | 0.0416233736013 | 0.0396004456328 | 0.0441770181991 | 0.0410926569719 | 0 / 0 / 0 / 0 |
| 15 | (0.5, 0.6, 0.0) | 0.0376096873389 | 0.0336989353178 | 0.0382507984294 | 0.0408793282695 | 13 / 13 / 0 / 0 |
| 16 | (0.5, 0.6, 0.05) | 0.0391809711543 | 0.036767258076 | 0.0406664181035 | 0.0401092372835 | 1 / 0 / 0 / 1 |
| 17 | (0.5, 0.6, 0.1) | 0.0421293072558 | 0.0398567898897 | 0.04520820037 | 0.0413229315076 | 0 / 0 / 0 / 0 |
| 18 | (0.6, 0.4, 0.0) | 0.0432368168995 | 0.0396930863848 | 0.0451232406311 | 0.0448941236828 | 0 / 0 / 0 / 0 |
| 19 | (0.6, 0.4, 0.05) | 0.0421819138651 | 0.0387512699235 | 0.0442265520338 | 0.043567919638 | 1 / 1 / 0 / 0 |
| 20 | (0.6, 0.4, 0.1) | 0.0436726925972 | 0.0404577530455 | 0.0471653622808 | 0.0433949624654 | 2 / 1 / 0 / 1 |
| 21 | (0.6, 0.5, 0.0) | 0.0411283857344 | 0.0399375289911 | 0.0383129099617 | 0.0451347182505 | 1 / 0 / 1 / 0 |
| 22 | (0.6, 0.5, 0.05) | 0.0412421674545 | 0.0390144561417 | 0.0408960122382 | 0.0438160339836 | 0 / 0 / 0 / 0 |
| 23 | (0.6, 0.5, 0.1) | 0.0432152933441 | 0.040694587538 | 0.0453270105645 | 0.0436242819298 | 0 / 0 / 0 / 0 |
| 24 | (0.6, 0.6, 0.0) | 0.0431123670191 | 0.0401386135956 | 0.0439750565449 | 0.0452234309167 | 0 / 0 / 0 / 0 |
| 25 | (0.6, 0.6, 0.05) | 0.0420627575212 | 0.0392115962692 | 0.0429949432611 | 0.0439817330334 | 0 / 0 / 0 / 0 |
| 26 | (0.6, 0.6, 0.1) | 0.0436052926273 | 0.0408703249879 | 0.0461200510385 | 0.0438255018555 | 1 / 0 / 0 / 1 |

## Gain over accepted hard Region2

Gain is Region2 MSE minus oracle-soft MSE; relative gain divides by Region2 MSE. Zero denominators stay null and are counted.

| Group | Absolute gain mean | Relative gain mean | Relative gain quantiles | Zero denominators |
|---|---:|---:|---|---:|
| spatial_pool | 0.00253587424134 | 0.0753885580102 | {'min': 0.0, 'p05': 0.0, 'p25': 0.0, 'p50': 0.007342507171290381, 'p75': 0.10418231769722702, 'p95': 0.33554802696181624, 'max': 0.5507384686867153} | 0 |
| left_right | 0.00128284627572 | 0.0426770261536 | {'min': 0.0, 'p05': 0.0, 'p25': 0.0017154493111369543, 'p50': 0.01207724966579769, 'p75': 0.07028392090101855, 'p95': 0.14131095779109096, 'max': 0.23935815280974213} | 0 |
| quadrants | 1.6612932086e-06 | 2.42373509236e-05 | {'min': 0.0, 'p05': 0.0, 'p25': 0.0, 'p50': 0.0, 'p75': 0.0, 'p95': 0.0, 'max': 0.0009694940369431687} | 0 |
| offset_left_right_40 | 0.0063231151551 | 0.183464410526 | {'min': 0.0, 'p05': 0.0, 'p25': 0.070364168543385, 'p50': 0.13630041495585735, 'p75': 0.2688221616699858, 'p95': 0.4920888397543115, 'max': 0.5507384686867153} | 0 |

## Literal interpretation

Threshold clauses: `{'oracle_improves_region2_5pct': True, 'oracle_improves_old_oracle_3pct': True, 'oracle_improves_best_fixed_3pct': True, 'fixed_improves_region2_5pct': False}`.

Strong adaptive-basis evidence requires all three oracle clauses. Fixed-continuous-basis evidence requires >=5% fixed-renderer improvement but <3% oracle advantage over that fixed renderer. Otherwise this screen is negative/inconclusive; none of these outcomes establishes that independently optimized soft actions are impossible.

Full 120×27 MSEs, fixed corner values/hashes, source-file hashes and per-episode gains are in candidate_metrics.json. Source/config hashes and the accepted data paths are in config.json. No candidate images were saved and accepted T015 files were not modified.

Stop for research-lead review. No reference-optimized actions, learned basis, new images or next experiment has been started.

## Execution, source evidence and verification

Run `20260913-000502-ttie-t016a-screen`, actual release `20260912-235801-ttie-t016a-screen`, exited0 at2026-09-12T16:05:23Z. A6000 physicalGPU1/logicalcuda0, Python3.12.12, Torch2.4.0+cu121/CUDA12.1, Pillow12.3.0. Source05e7dcf268c7b0479e6edabdc6ec76528d5730bf was committed/pushed before the diagnostic. The accepted provenance guard checked all61scientific files against that commit before reading data or creating outputs; local post-run verification independently checked their actual Git blobs.

Original T015 manifest SHA256 `77f405417324fc4e2d50ccd4606caa06725416c9b0f5182daac5dbdc067a4a09`; no new manifest or IDs. Original selected-grid tensors, output/trajectory file digests and source-metric hashes are retained per episode. The local verifier checked all120 saved physical corner grids against accepted trajectory snapshots at their already-selected steps, all3240candidate MSEs, fixed/oracle means, counts, gains/quantiles and literal clauses. Maximum aggregate recomputation difference1.3877787807814457e-17. No zero MSE denominators occur in these120rows; null handling is covered by the focused fixture.

Baseline5ISPtestsPASS2.759s; renderer4testsPASS.101s; final6focusedrenderer/metric testsPASS.060s; PythoncompilePASS. All120canonicalpixel/MSE discrepancies are exactlyzero, all3240identity checks and all saved-corner checks pass. No diagnostic implementation repair or scoring rerun was needed.

Raw execution archive: `a9330f26c3f2311572697bda33fb4b8b556573d9be0a4da52ecc02f465c86557`,69,856bytes, remote/local SHA match. Local raw/derived receipts are under `research_log/remote_runs/20260913-000502-ttie-t016a-screen/artifacts/audit`; actual server outputs are under `/home/wenchang/asdasdsad/wjq/TTIE/runs/20260913-000502-ttie-t016a-screen/artifacts/audit`. Accepted T015 pixels remain read-only at `/media/wenchang/F/wjq/TTIE/runs/20260912-213014-ttie-t015-fresh-ready/artifacts/audit`. No candidate image packs were generated. The local verification/runtime notes are derived evidence added after the raw archive snapshot.

Deployment issue preserved: remote shallow GitHubfetch failed withGnuTLS(-110) before any scoring. Exact local Git objects (onecommit,6015trees,61sourceblobs;1,244,916bytes, SHA2565e249b20217cf4df3cc5e9c110d98767aafb0b4aedb66fff264efaa00fcf50f2) were transferred into the new release. Full read-tree then attempted to fetch unnecessary historical artifact blobs; only that observed read-tree/HTTPS subprocess was stopped, with any lock retained. Populating the index from the exact61scientific entries allowed the unchanged provenance guard to pass61/61. Bookkeeping/artifact paths outside the allow-list may differ by design. This was metadata transport repair, not a guard bypass, scientific code change or experiment rerun. Runtime/source/data hashes are preserved in config.json, T016A_git_objects.json and T016A_runtime.json.

Reproduction: from a Git-backed checkout of05e7dcf, run `python -m ttie.soft_basis.screen --source-sha 05e7dcf268c7b0479e6edabdc6ec76528d5730bf --audit <accepted-T015-audit> --images <original-val2017> --manifest research_log/T015_manifest.json --output <new-diagnostic-output> --device cuda:0` using the recorded Torch/CUDA environment. This is a reproduction command, not an authorization to rerun now. The exact executed command is retained in run.sh/meta.json.

Recommended next decision: the predeclared strong screen passes, warranting research-lead consideration of a subsequent bounded boundary-adaptation/capacity audit. Neither independent action optimization nor learning/generalization has been tested here. Stop for the next explicitly issued task; do not train or extend the sweep automatically.
