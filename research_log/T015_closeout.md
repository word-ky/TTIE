# T015-CLOSEOUT — provenance binding and merge preparation

## 2026-09-12T15:13 heartbeat — accepted negative; closeout authorized

Research main `0b9adfc0320755abde53b787e4265ec6d1fd5597` accepts T015 as a controlled fresh negative and explicitly authorizes this engineering-only guard/tests closeout. No T016, fresh scoring, GPU experiment, or historical evidence rewrite. Main was merged into the engineering branch without conflicts; research-owned inbox/state preserved verbatim. The earlier decision to defer hardening is superseded only by this new explicit task.

Repair mode using the existing implementation skill. Baseline routing core: 2 tests PASS in 9.994s, log `T015_closeout_baseline.txt`. Existing full 132-test result and historical output verification are preserved. Inventory found no runtime-to-commit guard; existing receipt helpers validate old assets and remain unchanged. All current scientific runtime bytes equal their HEAD Git blobs before this patch.

Reuse map: retain all donor functions, trajectories, scoring/evaluation/metrics, historical receipts and manifests. Add only a standard-library Git byte-comparison helper, call it before pilot initialization, and test with tiny local Git repositories. Keep the helper in `ttie/routing/` so the accepted top-level donor inventory stays unchanged.

Increment 1: helper + real-Git fixture tests for correct/stale/invalid commits, byte differences, missing blobs/files, Git failure and dirty scientific index. Bookkeeping outside the explicit allow-list may differ. Compare actual runtime bytes directly to resolved commit blobs; return the hashes already verified for future config recording.

Increment 2 (after helper tests pass): integrate one guard at the Python entry point before model/data/output work. The shell wrapper will delegate directly to it; its old environment/test artifact prelude must not create output or run model fixtures before the guard. Full tests run separately for this closeout. Add entry-point ordering fixtures, then run the full local suite. No historical scientific outputs will be regenerated.

## 2026-09-12T15:21:43Z — helper and entry-point checks pass; full regression active

Increment1:9real-Git helper tests PASS12.659s. Increment2:11focused tests PASS26.891s, including real byte comparison before anyasset/model/evaluation/output work, and correct binding reaching assetpreflight. py_compile and GitBash syntax check PASS. Full unittest suite running locally, logT015_closeout_full_tests.txt (execsession21373). No scientificrunorhistoricaloutputregeneration. Future guard requires accessible Gitcommitobjects and rawbyte-identicalfiles; archive-only deployments without Git now fail closed instead of trusting a supplied SHA. Preparing future Git-backed deployment is outside this no-experiment closeout.

## 2026-09-12T15:24:04Z — full regression green

Full local unittest discover:143testsPASS118.624s, logT015_closeout_full_tests.txt. ExistingtinyCPUfixtures runonly intemporarydirectories; no COCOevaluation/CLIPexperiment/A6000job or accepted-outputregeneration. Explicitallow-list covers58/58existingdonor/routing/entrypoint files; no missingpaths. GitBash -n andpy_compilepass. HistoricalT015audit/manifest/receipt diffempty. Exactcodepatch: newprovenance.py,newtest_routing_provenance.py,pilotguard+verifiedinventory,thinrun_t015_a6000.sh. No renderer,objective,trajectory,router,metric,modelorassetchange. Committhisgreenpatchthenverifytheactualcommittedcheckoutwiththeguard.

## 2026-09-12T15:26:00Z — closeout verification complete

Guard implementation commit `a4041746301f3cca3550393d188f8461b7fe30a0` passes a read-only check of the actual checkout: all 58 allow-listed runtime files equal the declared Git commit blobs and the scientific worktree/index is clean. Supplying historical `c4e58e5ad64bfce0bea72561997db8007e12b510` to this newer checkout correctly fails at the changed pilot before any launch. Receipt: `T015_closeout_verification.json`.

Resolution of automatic P2 comment5646591797: one helper now resolves the declared commit, reads each Git blob, compares raw runtime bytes, and fails on mismatch/missing blob/missing file/invalid revision/Git failure. Scientific staged or unstaged changes fail; bookkeeping outside the allow-list is permitted. The Python entry point checks before RNG initialization, reading assets, loading models, scoring or creating outputs, and records the hashes it actually verified. The shell delegates directly and no longer creates environment/test artifacts or runs model fixtures first. There is no second validator or bypass flag. Unit tests are invoked separately with the existing command.

Exact code/test changes relative to closeout integration base `ab53c4b99cea86c61aa47d86ae2da83ea3d8df5f`:

- `ttie/routing/provenance.py`: new reusable helper and explicit 58-file list (original donor inventory, routing files including guard, T015 prepare/run scripts).
- `ttie/routing/pilot.py`: guard before initialization; use returned verified inventory for future config.
- `scripts/run_t015_a6000.sh`: thin direct entry-point delegation without pre-guard output/model/test work.
- `tests/test_routing_provenance.py`: 11 real-Git and entry-point tests.

Documentation/evidence: this closeout log, baseline/helper/focused/full test logs, real-checkout verification receipt, updated handoff and PR body. Historical evidence and the accepted source algorithms are untouched: diff of all historical `research_log/remote_runs`, T015 manifest/freeze audit and T014 receipt is empty. Scientific donor files are unchanged; no renderer, energy, trajectory, router or metric modification.

Commands and results: routing baseline2tests9.994s; provenance helper9tests12.659s; integratedfocused11tests26.891s; `D:\anaconda3\python.exe -m unittest discover -s tests -v`143tests118.624s, allPASS; `python -m py_compile` and `D:\Git\bin\bash.exe -n scripts/run_t015_a6000.sh`PASS. The full suite includes existing temporary CPU fixtures; accepted T015 outputs were not regenerated. No code fix was needed after these passing tests.

Historical T015 did **not** run with this guard. Research-lead main0b9adfc accepts its bounded negative based on separate retrospective7/7Gitblob evidence; timing/rawmetadata deviations remain disclosed. The new guard is explicitly authorized future engineering closeout. Future runtime directories must provide Git objects for the declared commit; archive-only copies without them now fail closed. No remote deployment or fresh experiment was attempted. Preparing future Git-backed deployments is outside this task.

Latest research-owned inbox/state from main0b9adfc were merged verbatim without conflicts. PR15 will remain unmerged for research-lead closeout acceptance. No T016/new basis/router training/detector/meta/prompt/ViT3 work follows this cycle.
