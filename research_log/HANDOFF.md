# TTIE handoff — T021-A PARTIAL, scientific binding pending

2026-09-13T16:10:37.440604+00:00

Draft PR41 https://github.com/word-ky/TTIE/pull/41 ; branch codex/T021A-ssim-audit. Preparation694cd5bd248f3b938660e88e12ca25f6876b495d; main PARTIAL report197646f6eb68cb6fff31f2d1a2b8372aabb95a50.

Need research-lead/user clarification of exact H0 method and primary bootstrap row pool BEFORE real SSIM evaluation. Accepted T014 has multiple controls, no unique H0 alias; fresh run has200 primary rows,80 heterogeneous subset and40 report-only offset rows. Question sent via main Codex outbox and async user input. Do not choose based on outcomes. This is not structurally unsupported; all frozen outputs are recoverable.

Completed:480 output/decision files,240 rows,40 images,3924677163 bytes SHA/size match accepted PR14 mergecebecffbd1335fade336df17d653eb4e5fb65ba3. No tensors/clean references loaded; no real SSIM/CI computed. Fixed full-RGB SSIM and10000 seed7 cluster-bootstrap implemented;2 tests pass1.01s. Full-map mean cross-check against skimage honors no-crop instruction; float64,11x11,sigma1.5,population,K1.01/K2.03,symmetric reflection. Bootstrap PCG64 and linear percentile CI fixed. Code/source: research_log/T021A_preparation.md, T021A_inputs/, T021A_outputs_verified.json, T021A_progress.md, ttie/ssim_transfer.py.

Original frozen images: /media/wenchang/F/wjq/TTIE/runs/20260912-191947-ttie-t014-stage-b/artifacts/audit/episodes/NNN/outputs.pt. Each has all controls/primary; exact clean reference can use identity image in same source's clean-condition episode (original unit gain/clamp). No new degrade/resize/render needed. First verify chosen outputs/receipt bindings before reading reference tensors. Prepare row triples and independent artifact-level SSIM replay after scope clarified. T014 sourcef861b2c6ffde6d017cb174ef8e00cb75701bf5e1.

Storage migration T012/T013/T014 Stage-A completed separately15:49:39Z; paths now symlink toF and all hashes preserved. This Stage-B output root was already onF and is unaffected. Preserve other remote jobs. No current job from this task.

Next heartbeat: read local handoff, fetch main, inspect protocol/state/both mailboxes and PR41 feedback. If baseline/pool newly specified, continue this existing task/PR; otherwise wait quietly, no duplicate recovery/report. Research-owned files untouched; no self-merge. No additional training, geometry repair or fresh cohorts.
