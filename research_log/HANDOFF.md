# TTIE handoff — 2026-09-11T23:19:21Z

T001–T005 ACCEPTED and merged. T006 DONE; PR https://github.com/word-ky/TTIE/pull/6 awaits research-lead review. Branch codex/T006-learned-exposure-prototypes; tested source5788335287232189335ee945c90ddafbeee6269e; evidence8c85cb6d301cfbc7ad9474566b580c218820de9f. Read current inbox/state/outbox and research_log/T006.md. No ISP adaptation/T007/CoOp authorized.

Fixed fresh100COCO manifest excludes all48prior IDs,60train/20calibration/20evaluation. Only three prototypes/1536scalars trained from900frozen CLIP features,500full-batch AdamW steps.46tests pass local/remote.700finite score rows; training CE .99505 -> .353822. Learned AUCdark.9870/bright.9366, homogeneouscorrectTPR93%/92%; mixedcorrectrecall91.25%/95%, wrong-type0%. Overall gate FAILS only clean FPR21%>15% (cleanquadrants20%). Strong discrimination does not waive failed clean-content requirement. Same-split zero-shot thresholds also calibrated on fresh clean split as predeclared; prompts/model unchanged. No post-hoc tuning.

Run20260912-071126-ttie-t006-a6000; release20260912-071122-ttie-t006; exit0 at2026-09-11T23:11:54Z. No active TTIE job. Full source-feature/train-history/learned-weight/calibration/score/localization receipts under research_log/remote_runs/<run-id> locally and TTIE/runs remotely. Learned prototypes.pt SHA256b4b32dbd96c65dcf606ee38d7450ebf348f5731823503b9c71ba15ec78217ac7. Model sameT004. Default artifact SFTP stalled; existing legacy SCP retry recovered all files, no retraining. T006_transfer_recovery.txt records issue.

Remote project /home/wenchang/asdasdsad/wjq/TTIE; freshimages shared/t006/images; model shared/t004. Same venv, no new deps. Use project .autodl/config.json and existing workflow scripts, explicit run IDs. Latest notes mirrored under remote root research_log outside tested release. Main mailbox worktree .autodl/main-mailbox; research-owned inbox/state read-only; Codex outbox append-only. Durable knowledge project-local.

Heartbeat ttie-chatgpt ACTIVE every15min. Await actionable feedback/new task; do not repeat completed T006 while OPEN persists.
