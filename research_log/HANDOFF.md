# TTIE handoff — T015 verified negative (4/10), awaiting research-lead review

T015 is complete. The one formal fresh evaluation and both remote/full-pixel and local/small-evidence verifications passed. Its scientific qualification is FALSE: 4/10 clauses pass. Do not rerun T015 because the research inbox still says OPEN. After evidence/mailbox delivery, wait for ChatGPT review or a new explicitly issued task. PR #15 must not be self-merged.

Read research_log/T015_analysis.md, T015.md, T015_spec.md, T015_freeze_audit.json, and the latest main coordination protocol/state/inbox/outbox. Engineering branch codex/T015-cross-basis-routing; PR https://github.com/word-ky/TTIE/pull/15. Scientific source c4e58e5ad64bfce0bea72561997db8007e12b510; manifest commit a22cab09e6a4dc31656688e9d6a742675146c333; published launch bookkeeping09546bfba2a1546dfab76391963c04e02003b3fc. No scientific code changed after freeze.

## Result and limits

40 fresh images x6 primary conditions =240 inputs. Spatial pool includes left_right, quadrants and offset_left_right_40. Clean mean/p95 and dark/bright clauses pass; six spatial comparison/non-inferiority/oracle-regret clauses fail. Routed spatial MSE0.037806382088456304; best fixed Region2 0.03504357374816512 (routing7.8839%worse); oracle among three already-selected outputs0.03436451945065831 (only1.9377%better than Region2, below required3%). Routed/oracle ratio1.1001574499751097; spatial disagreement51/120. This frozen candidate set lacks the declared aggregate oracle headroom, not just a successful raw-energy router. All688inspected IDs remain unavailable for corrective fresh evaluation. No training, score calibration, new basis, detector/meta/ViT3, or next task started.

## Exact execution and evidence locations

Formal run20260912-213014-ttie-t015-fresh-ready, exit0 at2026-09-12T14:00:40Z. Actual release20260912-212427-ttie-t015-fresh; physicalGPU1/logicalcuda0. There is no active T015 experiment or verifier now.

- Local project D:/work/fightccfa-agin/CVPR2027/TTT-ImageEnhancement.
- Original execution metadata/logs: /home/wenchang/asdasdsad/wjq/TTIE/runs/20260912-213014-ttie-t015-fresh-ready.
- All actual full image packs: /media/wenchang/F/wjq/TTIE/runs/20260912-213014-ttie-t015-fresh-ready/artifacts/audit. Default home/artifacts is empty. Keep all54,981,354,720image bytes and all earlier packs; no deletion.
- Complete local small evidence: research_log/remote_runs/20260912-213014-ttie-t015-fresh-ready/artifacts/audit. Includes output_verification.json, local_verification.json, final_distributions.md, margin_distributions.json, all raw metrics/small trajectories/receipts/six fixed figures.
- Complete smallarchive: local.autodl/T015_complete_small.tar.gz and remote homeTTIE/research_log/T015_complete_small.tar.gz; SHA2567b2ff9d718f946300c24872c60d132a34e9f0c3b80be2b308a28b4c3a159d9a6;30,288,221bytes. Only bank_images.pt/checkpoint_images.pt/outputs.pt omitted from download; original full packs remain F.
- Remote venv /home/wenchang/asdasdsad/wjq/TTIE/.venv. Official images homeTTIE/shared/t008/val2017; frozen CLIPshared/t004 and historicalT006assets unchanged.

## Verification complete

132localtests127.108s;132A6000startup tests42.521s; originalcalibrationbitwise andT014receipt pass beforefreshscoring. Full remote audit4560hashes,240inputs,23040energycheckpoints,22320updates,6002semanticcheckpoints,10589inactive-regionchecks: storedpixelsMSE/features/frozenenergies/checkpoints/routes/oracles/projections/summary exact. Local2400smallhashes/2400rows pass, maxsummarydifference6.938893903907228e-18. All6fixedID55167figures inspected and retained unchanged. No further tests/runs required absent substantive review feedback.

## Preserved deviations and recovery

Initial startup212626 failed before any fresh scoring because originalT006source_features.pt was missing. Copied exact historical file SHA114dab9df3760fff156a8295f9050dc07fb80e4d52b7308cce992c7df542d876, focused4tests passed, same frozen code/model/manifest then formal213014 passed. FailurearchiveSHA3604c24489e3f99ae5082bfcef6e9a8bb00da76b6d786d4d781473a93a7ac686 retained. Earlier tiny reporting failure and pre-upload packaging stop preserved in T015.md/analysis.

The requested separate pre-launch named-diff record was NOT saved before launch. T015_freeze_audit.json explicitly records a retrospective audit at2026-09-12T13:54:18Z. Actual frozen-to-launchdiff is only outbox/tasklog/manifest; seven startup scientific hashes equal frozen Gitblobs, donorinventory unchanged and committedmanifesthash matches. Preserve this documentation timing deviation; never backdate.

Rawmeta.releaseId wrongly says20260912-212421-taisp-t012-full because workflow last-release bookkeeping is shared across projects. Actual liveprocesscwd andTTIE/current both verifiedTTIE/releases/20260912-212427-ttie-t015-fresh; seven scientific hashes match. Preserve raw metadata and use freezeaudit for actual provenance. Leave unrelated workflow and other jobs unchanged.

Use only existing D:/work/claude-autodl/autodl-workflow-clean scripts with AUTODL_CONFIG_PATH=<localproject>/.autodl/config.json. Explicit run/release IDs are necessary because sharedworkflow last-run/last-release can refer to another project. For any future authorized deployment use ExtraExclude @('.autodl','research_log/remote_runs','*.tar.gz'); copy required original fixtures/checkpoints as recorded. Legacy SCP -O works; default SFTP closes connection. No credentials should be printed. No deployment is needed for T015 now.

## Collaboration

Existing heartbeat ttie-chatgpt is ACTIVE every15minutes on this task; ChatGPT hourly review was set by the user. Fetch main, inspect mailbox/PR feedback, act only on new/revised authorized tasks. If no actionable change, append only a local heartbeat check and stay quiet. Do not recreate automation. Keep current result/recovery notes in this project and mirror final notes to both remote project roots. Publish engineering evidence/readyPR15 and main Codex DONE outbox, then record exact delivery SHAs below. Engineering DONE does not mean scientific hypothesis passed or research-lead acceptance.

## 2026-09-12T14:37:42Z — evidence delivery
Engineering/evidence4b628702a38680cebaf39547887fa0ff1b8e7372. Fullreportanddistributions availableatthiscommit/PR15; mainCodexoutbox containsDONE4/10negative withall10clauses andevidencelinks. Finalrecoveryarchive51d9ae3de4d8703c0d9d0b4116cce9617a34958bd87ef7fd8309e2b91fcf2d08 verified/extractedbothserverroots, withderivedlocalverification/distributions alsocopiedintoactualF-audit. Awaitresearchleadreview/newtask; no activejob/duplicate.

Published mainDONE872f4f5215eb1c2fdb7fd1b180b3a57b116aaae7; PR15READY/unmerged. Exactdelivery IDs inT015_delivery.json. No further work until substantive review/newtask.
