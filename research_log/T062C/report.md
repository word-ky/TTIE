# T062-C — BLOCKED: prescribed cohort is not previously unused

Authorization: ce38725aa843cfd94aa192ca00aacc9945b8c61f. Tested cohort-preparation source: 21800caca4c0fe8c0e999e8b48fbcb389ab780e4. Branch: codex/T062C-fresh-qualification.

The exact prescribed selection has **0 overlap with T036 development100**, but **49/100 pairs have documented earlier reference use**. Of these, **18/100 are directly present in the accepted T032-A evaluation's actual normal-open receipt**. Therefore the literal selection rule and the task's previously-unused qualification requirement cannot both hold. This is a procedural BLOCKED result, not a scientific negative for step27.

The only cohort was selected from the accepted 689-pair train inventory, excluding exactly the T036100 low filenames, ranking the remaining589 by SHA256("T062C-v1:" + normalized relative low path), taking100 in hash order. Manifest was frozen at 2026-09-19T23:05:45.155830+00:00, SHA **935b545ce4c19b5441ac324195bf6152614783bca2415e0e6ec04fca9ccd68f3**, before the prior-use audit at 2026-09-19T23:05:45.351785+00:00. No image payload, new reference image, prior outcome metric, official-test data or cross-dataset data was accessed by this preparation. Existing JSON access ledgers are provenance metadata. No alternate cohort or expanded exclusion was used.

Actual T032 reference-use examples: Train/Normal/normal00616.png, normal00382.png, normal00368.png. Full49 list/reasons and18 direct receipt matches are in freshness_audit.json. Independent verify.py uses the older T032 train ledger, binary-digest ranking, and direct reference-open receipt to reproduce the exact100 selection, original100 exclusion and49/18 overlap counts; all source SHA bindings pass.

Validation: `python -B -m pytest research_log/T062C/test_prepare.py -q -p no:cacheprovider`: **2 passed in0.68s**. `python -B -m research_log.T062C.prepare`: sole cohort generation, exits0 persisting BLOCKED. `python -B -m research_log.T062C.verify`: PASS, exits0. Source was committed and pushed before cohort generation. PR136 comments/reviews were empty at this check.

No optimizer, renderer, gate/model, metric evaluation or GPU experiment was launched; no300-output freeze or efficacy numbers exist. Both A6000s were reachable with3501MiB free each; compute is available but cannot resolve the cohort contradiction. No implementation/model/threshold/stopping change. No changes to research-lead-owned files.

Requested lead decision: authorize a revised deterministic cohort rule excluding all previously reference-used training pairs using the accepted access ledger, or explicitly reclassify this exact cohort as a previously-exposed diagnostic. Do not silently call it fresh, tune step27, open official test or rerun the unchanged OPEN specification. Pending this decision, stop before adaptation/reference evaluation.
