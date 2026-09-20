# T062-C-R1 — DONE / COHORT_FREEZE_PASS

Authorization `ffe89df73bce4d30773b9401af3d92974a453cb3`. Tested/frozen source `08a6e869c5f7d5ff214d0a78bf0a3e950136d42e`. Branch `codex/T062CR1-unused-cohort`.

**689 Train pairs /416 ever-reference-used excluded /273 untouched remaining /100 selected.** Historical-reference overlap0; original T036 development overlap0. One exact selection by ascending SHA256("T062C-R1:" + normalized relative low path), with no alternate cohort or selection based on image quality, conditions, or method outcomes.

Manifest frozen at **2026-09-20T00:09:05.274043+00:00**, SHA256 **e8a66a350bcce5282df2e8114355afc434bcd0f4ea69e54957ebb7618f9183c2**. Independent verifier completed **2026-09-20T00:09:15.142252+00:00** and reproduced the exclusion union, full100 path order, per-path provenance ledger and exact serialized manifest hash from204 original commit/blob/SHA256-bound sources.

Coverage:77 preauthorization engineering branch tips,1526 task-owned JSON files scanned for explicit path-open/read fields,176 access-bearing records. Accepted T036 historical ledger excludes316; T036 actual cohort adds100. Identified later normal reads add no new Train pair beyond this union. T053/T056 records were recovered from their own branches. Explicit source/run/manifest mapping covers cases without dedicated path-read receipts, including T055-V's T022 split and T062-A/B's T036 split. Full rationale and scope are in COVERAGE.md. Historical dependency digests verified; one previously documented T023 receipt uses its accepted CRLF representation.

Selection consumes only a separately hashed path/boolean projection. The committed allow-list binds all204 original provenance/review files. The complete canonical ever_reference_open ledger stores every Train pair's boolean and source reason IDs. No missing/mismatched provenance or unresolved historical Train access was found in this audit.

Commands: `python -B -m research_log.T062CR1.discover`, `...bind`, `...prepare`, `...verify`. Final tests: `python -B -m pytest --import-mode=importlib research_log/T062CR1/test_prepare.py research_log/T062C/test_prepare.py -q -p no:cacheprovider`: **5 passed in0.36s**. AST syntax checks passed. Prepare and independent verify each exit0. Source committed/pushed before sole cohort generation. Independent verifier imports no selector and uses binary-digest heap selection.

Observed preparation failure: combined pytest collection initially collided on two test_prepare.py basenames (1 collection error); rerunning with pytest's importlib mode fixed collection without changing code or expectations. Earlier focused3tests passed0.70s. No provenance, selection, image-access or verifier failure occurred; no second cohort.

**Normal/reference image opens0; GPU/model/optimizer/render/evaluation runs0; PSNR/SSIM computations0; official-test/cross-dataset image access0.** This PASS establishes provenance/cohort readiness only. It does not establish T062 step27 efficacy or generalization. Method/loss/step27/thresholds untouched. Stop and wait for the next research-lead cycle; do not run qualification without the next task.
