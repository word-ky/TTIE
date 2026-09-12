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
