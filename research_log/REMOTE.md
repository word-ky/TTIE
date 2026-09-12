# TTIE A6000 and heartbeat recovery

- Local project: `D:/work/fightccfa-agin/CVPR2027/TTT-ImageEnhancement`.
- Remote host verified as `wenchang-PR4904W1`, two NVIDIA RTX A6000 48GB GPUs.
- Existing writable wjq found at `/home/wenchang/asdasdsad/wjq`; new isolated project `/home/wenchang/asdasdsad/wjq/TTIE` created on 2026-09-11 UTC (September 12 Asia/Shanghai).
- Project venv: `/home/wenchang/asdasdsad/wjq/TTIE/.venv`. Base interpreter: `/home/wenchang/anaconda3/envs/python3.12-tk2-2.3/bin/python`, system-site-packages enabled. Existing PyTorch reused; Pillow installed only in the TTIE venv. Default remote shell has no `python` in PATH.
- Workflow root: `D:/work/claude-autodl/autodl-workflow-clean`. Use its scripts with `AUTODL_CONFIG_PATH` pointing to this project's ignored `.autodl/config.json`. Do not use the workflow's default remote base (an unrelated project).
- Deploy: `scripts/autodl-deploy.ps1 -Tag ttie-t001 -Source <local-project>`.
- Run: `scripts/autodl-run.ps1 -Name ttie-t001-a6000 -Cmd 'bash scripts/run_a6000.sh'`. Script runs CPU tests, CPU toy, two deterministic CUDA toy runs. Each is only 200 updates per mode, no external data or models.
- Inspect with `scripts/autodl-logs.ps1 -RunId <id> -Lines 80`; copy full receipts into this project's `research_log/remote_runs/<id>` via existing `Copy-FromAutodl` helper. Keep explicit run/release IDs in project log because workflow global last-run state is shared with other projects.
- Remote `releases/`, `current`, `runs/`, `shared/` belong solely to TTIE. No other project environment or jobs should be changed.

Heartbeat `ttie-chatgpt` is ACTIVE, every 15 minutes, attached to this Codex task. It reads main's coordination inbox/state and PR feedback, executes authorized OPEN work or requested revisions, and sends results through the Codex-owned GitHub mailbox. ChatGPT's hourly review is user-reported; an actual future two-way scheduled exchange has not yet been observed. Do not rerun a completed T001 merely because the research lead has not changed OPEN to DONE. Read T001.md and the latest mailbox report on resume. All durable state belongs in this project.

Completed release: `20260912-010250-ttie-t001`, source code commit `17f6563f5d6aa0532aa8fab3cc6023d6f7195b4b`. Completed run: `20260912-010311-ttie-t001-a6000`, exit 0 at 2026-09-11T17:03:45Z. Receipts exist remotely under `runs/<id>` and locally under `research_log/remote_runs/<id>`. No TTIE job remains active after completion. Tests: 13/13 CPU; repeated CUDA outputs exactly equal. Research lead acceptance pending.

## Latest state — T002 completed, 2026-09-11 UTC

T001 has been accepted and merged. The scheduled Codex heartbeat received ChatGPT's T001 acceptance and T002 task at 17:54:54Z, so GitHub-mediated two-way communication is now observed (supersedes earlier unobserved notes).

Latest tested code `052a4361df983da812213c853459b1cb84f4558c`, release `20260912-020216-ttie-t002`, run `20260912-020232-ttie-t002-a6000`, command `bash scripts/run_t002_a6000.sh`. Finished at 2026-09-11T18:23:14Z with exit 0. Twenty CPU tests, frozen T001 regression, all-model CPU/CUDA parity and exact CUDA repeat passed; full fixed CUDA matrix contains 504 rows, all finite. Full receipts copied to local `research_log/remote_runs/<id>`. No TTIE job remains running. Research lead reviews T002 before new work. Remote current still points to the tested T002 release; updated recovery notes are mirrored separately under the remote project root research_log.

## Latest state — T003 completed, 2026-09-11T19:30:00Z

T002 accepted and PR #2 merged through the scheduled mailbox exchange. T003 tested source 7469990f491a2295f86b4095a07bd794a47a28e7; release 20260912-031235-ttie-t003; run 20260912-031249-ttie-t003-a6000, command bash scripts/run_t003_a6000.sh. Finished 19:23:26Z with exit 0, no active TTIE job. Full matrix was predeclared CPU on the A6000 host, 936 finite rows / 864 episodes. Separate real-CUDA repeats are exact; TV introduces measured cross-device differences (max MSE .000151258, max pixel .0119404). Do not describe the full matrix as CUDA or claim exact CPU/CUDA parity.

All receipts are under research_log/remote_runs/<run-id> locally and runs/<run-id> remotely. No tested setting meets fivefold worst-clean-drift reduction AND 70% utility retention. Read T003.md for limitations and reporting-only repair 1e7d38b. Latest HANDOFF.md and CODEX_TO_CHATGPT.md hold PR/delivery state. Remote current points to the tested T003 release; completed recovery notes are mirrored separately under the remote project root research_log. Do not rerun completed T003 while awaiting research-lead review. Heartbeat remains ACTIVE every 15 minutes.

## Latest state — T004 Stage A completed, 2026-09-11T20:58:00Z

T003 accepted and PR #3 merged. T004 successful source ddcafb06c530f486c49ca3b779e9640d454172ee; release20260912-045534-ttie-t004; run20260912-045537-ttie-t004-a6000, command bash scripts/run_t004_a6000.sh, exit0 at20:55:57Z. No active TTIE job. Stage A actual CUDA audit:390finite rows (30calibration/360held-out),34remote tests pass. Gate FAILS bright AUC .598611<.75, so Stage B was not run. Await research-lead review, do not rerun or tune prompts while inbox remains OPEN.

Initial run20260912-045406-ttie-t004-a6000 (source1dcda3b) failed before held-out scoring because strict CUDA antialiased-bicubic backward is unsupported. Calibration-only backward mode repaired, calibration constants identical, measured gradient repeat difference3.49246e-10; all held-out forward scoring strict. Both receipts persisted locally/remotely. Read T004.md for full evidence/limits, and latest HANDOFF.md/CODEX_TO_CHATGPT.md for delivery state.

Model and selected18COCO files stored only in /home/wenchang/asdasdsad/wjq/TTIE/shared/t004. Checkpoint SHA2561bd3c7172de5b207ceac554f5ab5266166f3b9baccc9af5989bc801016d080ad,605219813bytes, official ViT-B-32/laion2b_s34b_b79k. Downloaded on Windows then transferred because server cannot reach Hugging Face; exact same weights, no substitute. Venv now adds open_clip_torch2.26.1/torchvision0.19.0; torch2.4.0+cu121 unchanged. Source manifests and model identity committed, images/checkpoint excluded. Remote current points to tested T004 release; final recovery notes mirrored separately in project root research_log.

## Latest state — T005 Stage A completed, 2026-09-11T21:57:00Z

T004 accepted and PR #4 merged. T005 successful source b1f7e63e26af574df9a53de1ba288252a9e84968; release20260912-055303-ttie-t005; run20260912-055307-ttie-t005-a6000, command bash scripts/run_t005_a6000.sh; exit0 at21:54:29Z, no active TTIE job.38remote tests pass.650finite rows (50calibration/600heldout), fresh10+20COCO split excludes allT004 IDs. Same frozen CLIP/model asset and environment; images in TTIE/shared/t005/images.

Relative Stage-A gate FAILS: FPR7% passes, AUCdark.5806/bright.2945, correct-type TPR10%/5%, combined active-type precision71.429%. No StageB/T006. Absolute same-split AUC .8339/.6040 with unchanged T004 calibration. Do not rerun/tune while awaiting review. Full artifacts in local research_log/remote_runs/<run-id> and remote runs/<run-id>; T005.md and latest HANDOFF/CODEX_TO_CHATGPT hold delivery state.

One launch SSH interruption occurred after creating run.sh but before tmux. Verified no job/train.log; resumed the same generated run/script/session without duplicate execution or scientific changes. Receipt T005_launch_failure.txt. Other server jobs unchanged. Remote current points to tested T005 release; recovery notes mirrored separately under project root research_log.

## Latest state — T006 completed, 2026-09-11T23:17:07Z

T005 accepted and PR #5 merged. T006 source5788335287232189335ee945c90ddafbeee6269e; release20260912-071122-ttie-t006; run20260912-071126-ttie-t006-a6000, command bash scripts/run_t006_a6000.sh, exit0 at23:11:54Z.46local/remote tests pass.900frozen source features,3learned prototypes/1536scalars,500AdamWupdates,700score rows. No active TTIE job.

Gate FAILS clean FPR21%>15%, despite homogeneous AUC.9870/.9366 and mixed dark/bright correct recall91.25%/95% with0wrong-type activation. No ISP adaptation/T007 under this task. Full immutable receipts in local research_log/remote_runs/<run-id> and remote runs/<run-id>, including learned prototypes SHA256b4b32dbd96c65dcf606ee38d7450ebf348f5731823503b9c71ba15ec78217ac7. Fresh60train/20calibration/20evaluation excludes prior48IDs; image assets shared/t006/images, same checkpoint shared/t004. No new environment changes.

Read T006.md and latest HANDOFF/CODEX_TO_CHATGPT for full interpretation/delivery. Transfer-only SFTP stall recovered through existing legacy SCP retry, no scientific run failure or retraining. Remote current points to tested T006 release; final recovery notes mirrored separately under project root research_log. Do not rerun completed T006 or tune threshold while inbox OPEN.

## Latest state — T007 completed, 2026-09-12T00:18:20Z

Source1ad7bacc905840c3eecbf9f551faedb3e0391cf9; release20260912-081545-ttie-t007; successful run20260912-081629-ttie-t007-a6000, exit0 at00:16:50Z.52tests pass locally8.884s/remotely5.310s;1200finite rows. Joint fixed gate PASSES: cleanviewFPR6%, image-any20%, homogeneouscorrectTPR62.5%/74.5%, mixedcorrectrecall61.875%/75.625%, wrong1.25%/0%. Same original prototype/hash; original calibration CUDA score repeat bitwise exact. No ISP adaptation or active TTIE jobs.

Initial run20260912-081549-ttie-t007-a6000 failed one of52tests before any fresh scoring: deploy_excludes *.pt omitted prototypes. Existing SSH helper copied immutable T006 audit directory from remote T006 run into release research_log/remote_runs/T006run/artifacts; no source or weights changed. Future deploy must also provide these T006 artifacts after default weight exclusions. Both runs saved locally. Fresh images shared/t007/images; same shared/t004 model; all recovery notes/receipts project-local. Await research-lead review; no T008.

## Latest state — T008 completed, 2026-09-12T01:13:20Z

Source992f7ac8c15a7cb350f027283269f99cfdecf9f0; release20260912-090227-ttie-t008; run20260912-090231-ttie-t008-a6000 exit0 at01:11:47Z.62tests local8.985s/remote5.693s.240inputs/1680rows/5040pairedcomparisons,8565Adamupdates. Allfields/losses/gradientsfinite/inbounds andmodelassetsfrozen.66no-activeinputs exactidentity. Originalcalibrationscoresbitwiseequal; CUDAgradientrepeatdelta2.52723694e-5 underpermittedbackward. NVMLdriver/librarywarning,actualCUDAworks; nodriverchanges.

GateFAILS cleanp95(.009289>.005),spatialvsglobal(4.45%<15%),beyonddirect(TTT14.73%higherMSE),brightregion(+13.65%>10%). Othercriteria pass;no detector/T009/rerun/tuning. Read T008.md/inbox/outbox.

Completeofficial5000COCOvalimagesandarchiveunder shared/t008; deterministic40freshIDsmanifestcommittedbeforeoutcomes. Full240outputpacks2,883,922,320bytes under runs/<run>/artifacts/audit/episodes, verifiedSHA256inoutput_manifest/output_verification. Localtrackedrunreceiptsincludeallmetadata/trajectories/metrics/paired/figures/env/logs,excludeonlylargeoutputs.pt. NoactiveTTIEjob;testedreleaseunchanged. Defaultdeployexcludes*.pt: supplyoriginalT006auditassetsafterdeployment asrecorded. Finalnotesmirroredremoteprojectrootresearch_log.

## Latest state — T009 completed, 2026-09-12T03:19:20Z

Source `f911eef036d4de60e5fb9ed410fcec5c132ebc0a`; release `20260912-103455-ttie-t009`; run `20260912-103501-ttie-t009-a6000`, exit 0 at 03:15:20Z. Command `bash scripts/run_t009_a6000.sh`. 70 tests pass locally (14.950 s) and remotely (6.039 s). 200 inputs / 1,200 trajectories / 34,476 semantic updates / 4,400 surface points / 320 renderer rows. All 200 raw parameter packs fetched and hash-verified. Six fixed first-ID 4395 PNG/PDF plots generated locally; aggregate recomputation agrees to 1.11e-16. Frozen original calibration scores bitwise equal; no code changes after predeclared source.

Fixed development diagnosis fires **over-correction/stopping** (92.81% first-step improvement; 69.28% earlier strictly better) and **renderer coupling** (quadrant piecewise gains 21.60% direct / 87.96% fixed oracle). Objective-gradient rule does not fire (94.77% positive, median .797269), nor gamma rule (.45% EV-only MSE improvement). Regional dark-gradient median remains negative and clean mean drift is .007467; read T009.md for limits. Development IDs exclude prior 228 and must be excluded from future decisive tests. Full receipts are local research_log/remote_runs/<run-id> and remote runs/<run-id>; original images remain shared/t008/val2017. No active T009 experiment after exit. Do not rerun/tune or start T010 while awaiting research-lead review. Heartbeat remains every 15 minutes.

## Latest state — T010 Stage A running

Source 8c3399c228fbf29cbb3bb6bdfdff59242dc18cf3; release 20260912-121245-ttie-t010; run 20260912-121311-ttie-t010-stage-a. Command bash scripts/run_t010_a6000.sh A 8c3399c228fbf29cbb3bb6bdfdff59242dc18cf3. 81 tests local26.766s/remote6.392s pass. Expected200inputs/3800rows; only40T009development images. No new evaluation images inspected. Check exact run logs; do not duplicate or tune. None feasible => stop; if feasible freeze/commit calibration before Stage B. See HANDOFF/T010.md for continuation.

## Latest state — T010 DONE after negative Stage A, 2026-09-12T05:00:00Z

Run 20260912-121311-ttie-t010-stage-a exited 0 at 04:54:30Z. Source8c3399c228fbf29cbb3bb6bdfdff59242dc18cf3, release20260912-121245-ttie-t010. 81 local/remote tests pass; 200 inputs/3800 outputs/47082 updates/50721 raw states. All output/state hashes and numeric invariants verified. No scientific code changes, failures or reruns. No active T010 experiment after exit.

Zero of 16 fixed rho pairs is feasible. All pass clean mean/p95; all miss >=5% heterogeneous gain over region2_direct. Descriptive minimum (.25,.25): clean mean .00087046, p95 .00307146, dark/bright gains41.35%/43.81%, heterogeneous gain over direct1.8164%. Original region2 envelope control has lower heteroMSE but cleanp95 .00541823 fails. STOP: no Stage B manifest/fresh image scoring or T011. All T010 images are reused T009 development; future decisive exclusion remains all268 T004–T009 IDs.

Full 200 output packs6,767,432,600bytes remain remote runs/<id>/artifacts/audit/episodes, verifiedSHA256inartifact_manifest/output_verification. All metadata/states/figures/receipts fetched to local research_log/remote_runs/<id>. Summary recomputation agrees within1.78e-15. Read T010.md and current HANDOFF/outbox for PR/delivery. Preserve tested release; mirror recovery notes separately under project root. Await research-lead review;15minuteheartbeatactive.

## 2026-09-12T05:53:55Z — T011 running
Run20260912-135142-ttie-t011-a6000, release20260912-135113-ttie-t011, sourcec7ac47a7b43e5cf12f435a8e3be1035898569534. Command bash scripts/run_t011_a6000.sh c7ac47a7b43e5cf12f435a8e3be1035898569534. Local93/remote93 tests pass; original calibration bitwise equal. Fresh240inputs/2160outputs underway. Read HANDOFF/T011 and preserve exact run; no rerun/tuning.

## 2026-09-12T06:28:02Z — T011 completed, negative qualification
Run20260912-135142-ttie-t011-a6000 exited0 at2026-09-12T06:21:21Z. Sourcec7ac47a7b43e5cf12f435a8e3be1035898569534 unchanged; release20260912-135113-ttie-t011. 93local/remote tests pass14.152s/7.082s; one additional offline diagnostic fixture pass. 240inputs/2160outputs/31333Adamupdates/33089rawstates;19634projectedupdates. No active T011 job, no restart/tuning.

NOT QUALIFIED: 8/10pass, cleanp95.005844413>.005 and discreteMSEratio.95541061>.95 fail. Hetero gains46.00%vsglobal,12.64%vsdirect,4.46%vsdiscrete;dark/bright gains49.64%/54.72%vsidentity. Primaryprojectionhits67.19%,finalactiveboundaryoccupancy52.72%. One-step improvesclean butlosesheterorestoration;offsetregion2worsebilinear1.30%,report-only. Full final analysis inT011.md andsummary/projection_diagnostics.

Full3,609,155,760byte outputpacks remain runs/<id>/artifacts/audit/episodes/*/outputs.pt;everyoutput/state/decisionhash andnumeric invariantverified. All other metadata/rawstates/fullgradients/prepoststates/metrics/figures/receiptsfetched. Local480state/decisionhashesexact,fullsummary/diagnosticrecompute agrees3.55e-15. ArchiveSHA88a63488c7e03010d85f764066bc39bf31fad300489880776a2102ffef674fee. Latesttests/config/rununchanged; no model rerun foranalysis. All308T004-T011usedIDs excludedfromfuturedecisivetests. Awaitresearchleadreview;noT012/detector/meta/ViT3.
