# T015 — frozen cross-basis routing: bounded fresh negative result

The completed 40-image × 6-condition experiment passes **4 of 10** predeclared clauses. Frozen raw-energy routing does not establish the requested adaptive-basis benefit. On the 120-input spatial pool, routed MSE is 0.037806382088456304 versus the best fixed basis, Region2, at 0.03504357374816512: routing is 7.8839% worse. The evaluation-only oracle among the three already-selected outputs reaches 0.03436451945065831, only 1.9377% better than Region2. Even this oracle misses the required 3% aggregate improvement.

These results concern the three frozen selected outputs and this synthetic protocol. They do not prove that every spatial parameterization lacks capacity. They also do not justify treating energy calibration as the sole remaining issue: the current candidates themselves provide insufficient oracle headroom for the declared aggregate claim. Preserve this negative result and await the research lead's decision. No calibration, refitting, new basis, additional fresh split, or next task has been started.

## Frozen execution

- Scientific implementation: `c4e58e5ad64bfce0bea72561997db8007e12b510`.
- Manifest commit: `a22cab09e6a4dc31656688e9d6a742675146c333`; SHA256 `77f405417324fc4e2d50ccd4606caa06725416c9b0f5182daac5dbdc067a4a09`.
- Published pre-launch bookkeeping commit: `09546bfba2a1546dfab76391963c04e02003b3fc`.
- Actual release: `20260912-212427-ttie-t015-fresh`.
- Formal run: `20260912-213014-ttie-t015-fresh-ready`, exit 0 at `2026-09-12T14:00:40Z`, A6000 physical GPU 1.
- Frozen T014 Sobolev head SHA256: `c3d1eef9f20af163e1268823cf16d3fdaed315e7723fc94db3b1138ad1336521`; original receipt SHA256 `c6c611aa5769d9f857ff57675e745abe8415721c0315458b85d30e1b5b5c9a68`.
- Exactly 40 official COCO val2017 images, excluding 648 previously inspected/source IDs; original shorter side >=320; no annotation access. All 688 inspected IDs now remain unavailable for corrective fresh evaluation.
- Six primary conditions: clean, homogeneous dark, homogeneous bright, left/right, quadrants, offset left/right 40%. Spatial pool includes all three heterogeneous conditions; aligned heterogeneous includes left/right and quadrants only.

The router receives exactly three selected scalar energies. It applies literal argmin and exact tie order global → bilinear2 → Region2. Each candidate reuses the frozen T014 trajectory with identity reset, Adam 0.03, 40 updates when active, and minimum-energy checkpoint selection with earliest ties. Routing copies an already-selected output; it creates no fourth trajectory. Training steps are zero. Output, trajectory, score, routing decision and hash persistence precede reference access. The reference-only oracle selects among the three selected outputs, not arbitrary trajectory checkpoints.

## All ten predeclared clauses

| Clause | Observed value or MSE ratio | Required | Result |
|---|---:|---:|---|
| Clean mean MSE | 0.00026011993759311736 | <=0.003 | PASS |
| Clean p95 MSE | 0.000014184007886796383 | <=0.005 | PASS |
| Dark / identity | 0.3834196586784088 | <=0.60 | PASS |
| Bright / identity | 0.37329369013445 | <=0.60 | PASS |
| Spatial / best fixed basis | 1.0788392291307287 | <=0.97 | FAIL |
| Spatial / projected discrete | 1.0321708677502732 | <=0.95 | FAIL |
| Spatial / frozen semantic step16 | 1.0336574712399433 | <=0.95 | FAIL |
| Offset / bilinear2 | 1.0329970892024536 | <=1.01 | FAIL |
| Aligned heterogeneous / Region2 | 1.1098498897633413 | <=1.01 | FAIL |
| Spatial / oracle best basis | 1.1001574499751097 | <=1.05 | FAIL |

The literal conjunction is false. The generated `final_distributions.md` in the completed run audit gives all absolute MSE thresholds, full method tables, every condition's routing/oracle counts, complete margin quantiles, conditional regret, trajectory step histograms, projection and boundary fractions. `routing_diagnostics.json` preserves every raw score and margin.

## Routing and available headroom

| Condition | Routed global / bilinear2 / Region2 | Oracle global / bilinear2 / Region2 | Disagreements |
|---|---|---|---|
| Clean | 38 / 1 / 1 | 39 / 1 / 0 | 2/40 |
| Dark | 32 / 5 / 3 | 39 / 1 / 0 | 8/40 |
| Bright | 12 / 12 / 16 | 33 / 6 / 1 | 23/40 |
| Left/right | 6 / 18 / 16 | 5 / 1 / 34 | 25/40 |
| Quadrants | 1 / 10 / 29 | 3 / 0 / 37 | 12/40 |
| Offset | 8 / 14 / 18 | 8 / 18 / 14 | 14/40 |
| Spatial pool | 15 / 42 / 63 | 16 / 19 / 85 | 51/120 |
| All | 97 / 60 / 83 | 127 / 27 / 86 | 84/240 |

Spatial routing/oracle disagreement is 42.5%, with routed/oracle MSE ratio 1.1001574499751097. Conditional spatial mean excess MSE is 0.007827205831805866 for routed global (15 cases), 0.006070732538189206 for bilinear2 (42), and 0.0006451533722972113 for Region2 (63). Corresponding ratios of mean MSE are 1.197118795162942, 1.2023952305968049 and 1.0179181247965847. Disagreement counts can include equal-output ties; absolute regret is the relevant consequence.

Oracle/best-fixed spatial MSE is **0.980622572846402**, above the required 0.97. Offset oracle/bilinear2 is 0.9710560142767047, showing some local complementary candidates but no qualifying aggregate headroom. Spatial winner–runner-up raw energy margin mean/median/p95 is 0.08359103798866271 / 0.055542945861816406 / 0.25876606702804567. Scores remain unnormalized and unadjusted.

There are 38 clean zero-oracle cases, all with zero absolute regret; undefined ratios remain null rather than silently divided by zero. There are no spatial zero-oracle cases and no positive-regret zero-oracle cases. Clean mean/p95 qualification is an aggregate result, not a worst-case preservation guarantee.

Each basis has 54 inactive and 186 active episodes, 7,440 updates, yielding 23,040 total energy checkpoints and 22,320 updates. Global/bilinear2/Region2 projection fractions are 4281/7440, 6932/7440 and 6972/7440; movable-boundary fractions are 199/372, 619/982 and 605/982. The resulting trajectories remain strongly constrained by the existing action box. This is a recorded property, not grounds to modify it after evaluation.

## Verification and evidence

Baseline: 125 local tests passed in 95.413s. Focused router tests: 2 passed in 5.865s; metrics: 3 in 0.406s; final tiny six-condition integration: 2 in 16.284s. Final full suite: **132 local tests in 127.108s**, **132 A6000 startup tests in 42.521s**. Before fresh scoring, the original calibration was bitwise equal and the frozen T014 receipt verified.

`verify_t015.py` completed against all saved remote pixels and traces: 4,560 hashed files, 54,981,354,720 large-image bytes, 240 inputs, 23,040 energy checkpoints, 6,002 semantic checkpoints and 10,589 inactive-region checks. Stored-pixel MSE, frozen energies, selected checkpoints, routes, oracles, features, projections and summary all match. The audit confirms no refitting, immutable assets/manifest and offset as a primary condition.

Small evidence archive SHA256: `7b2ff9d718f946300c24872c60d132a34e9f0c3b80be2b308a28b4c3a159d9a6`, 30,288,221 bytes. It includes all small traces, metrics, receipts, six fixed example figures and original execution metadata/logs. Large pixel packs remain at `/media/wenchang/F/wjq/TTIE/runs/20260912-213014-ttie-t015-fresh-ready/artifacts/audit`; prior packs remain intact. Local extraction lives under `research_log/remote_runs/20260912-213014-ttie-t015-fresh-ready/`. The independent local verification receipt and generated distributions are added alongside the immutable raw evidence after download.

Commands: `CUDA_VISIBLE_DEVICES=1 bash scripts/run_t015_a6000.sh c4e58e5ad64bfce0bea72561997db8007e12b510` with `AUTODL_ARTIFACTS_DIR` pointing to the F run; remote `verify_t015.py <audit> --images <shared/t008/val2017>` under the frozen release and deterministic CUDA environment; local `verify_t015_local.py <audit> --archive .autodl/T015_complete_small.tar.gz`; reporting-only `summarize_t015.py <audit>`.

## Preserved failures and deviations

1. The initial tiny integration reducer assumed T014's absent value-only control and raised KeyError. Only the T015 reporting reducer was corrected before the scientific freeze; the old donor code remained unchanged. The original failure is preserved in `T015_pilot_initial_failure.txt`.
2. Initial metadata packaging included historical archives. The observed local tar process was stopped before upload; the partial archive and empty release were retained. Subsequent deployment explicitly excluded `.autodl`, `research_log/remote_runs` and `*.tar.gz`.
3. Startup run `20260912-212626-ttie-t015-fresh` failed one of 132 tests because the historical T006 `source_features.pt` fixture was missing. No preflight or fresh scoring had occurred. The unchanged original fixture was copied, its SHA verified, four focused tests passed, and the same frozen release/manifest passed all 132 tests in the formal run. Failure archive SHA256: `3604c24489e3f99ae5082bfcef6e9a8bb00da76b6d786d4d781473a93a7ac686`. This was a pre-scoring deployment repair, not an outcome-driven scientific rerun.
4. Two transient Git HTTPS failures and an SSH copy timeout recovered before launch; publication of the frozen manifest and launch bookkeeping was confirmed before fresh scoring.
5. The interim review requested a separate pre-launch named-diff record. That record was not saved before launch. `T015_freeze_audit.json` is explicitly retrospective at `2026-09-12T13:54:18Z`, never backdated. The actual diff from frozen implementation to published launch commit contains only outbox, task log and manifest. All seven startup scientific code hashes equal frozen Git blobs; the original T014 inventory and committed manifest also match. This supports immutability while preserving the documentation-timing noncompliance.
6. Raw `meta.json.releaseId` incorrectly says `20260912-212421-taisp-t012-full`. The workflow uses shared last-release bookkeeping across projects. The live TTIE process cwd and TTIE/current were both verified as the actual `20260912-212427-ttie-t015-fresh` release; seven startup scientific hashes match the frozen source. Raw metadata is preserved, not rewritten; `T015_freeze_audit.json` records the correction. The unrelated workflow and other jobs were left unchanged.

Engineering completion and scientific qualification are separate: T015 is ready for review as a completed negative result. ChatGPT owns acceptance and the next research task; PR #15 must not be self-merged.

## Final local verification and fixed visual inspection

The completed archive downloaded successfully and its SHA256 exactly matches the remote archive. Local verification passes 2,400 small-file hashes and 2,400 evaluation rows; independent summary/diagnostic recomputation has maximum absolute difference 6.938893903907228e-18. Frozen head and manifest match. Full reporting-only distributions were generated successfully; the spatial energy margin has seven exact zeros out of 120.

All six fixed first-manifest-image panels (ID 55167) were visually inspected. Layouts contain the clean reference, degraded identity and all methods/oracle with readable method labels. The clean example is preserved; the dark example remains visibly dark across methods, and heterogeneous examples retain residual exposure discontinuities. These fixed examples are retained without favorable selection or visual editing; aggregate metrics, not these six panels, determine the negative verdict.