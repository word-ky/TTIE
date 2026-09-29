## 2026-09-29 19:00 +08 — UHD-LL frozen outputs still need A6000 preservation

Correction to earlier shutdown advice: the paid host remains reachable, but only SMID/LSRW/SID migration was verified; several accepted UHD-LL outputs remain on its instance-local root overlay. A non-destructive persistent archive of PromptIR static, PromptIR+DCTTA, Ours-Step0, Ours-TTT-abstention and DCTTA adapted state is **in progress** as paid run `20260929-185614-ttie-uhdll-ephemeral-preserve` (`/autodl-fs/data/TTIE/migration-20260929-UHDLL-ephemeral.tar`). It is not accepted until exit 0, hashes, A6000 transfer/extraction and counts pass. RetinexFormer/SNR-Aware 150+150 `.npy` outputs are already on paid persistent storage but also need A6000 transfer. The existing SSH-key `scp -3` relay path passed a small-file SHA test. **Keep paid host open until the complete transfer is verified.** No UHD-LL reference access or new GPU inference. Details in `research_log/migration_20260929_A6000.md`. A6000 SMID ceiling round 1 and its finite continuation remain separate, GPU0-only.

---

## 2026-09-29 18:36 +08 — finite GPU0 ceiling continuation queued

The A6000 run `20260929-183549-ttie-remaining-ceilings-gpu0` now waits for SMID round 1's exact success marker, then executes the already verified SMID finish, SID round 1, and SID finish runners **sequentially on physical GPU0 only**. Its source is `research_log/migration_20260929_A6000/run_remaining_ceilings.sh`, SHA256 `34ba54f46b773906b1599a35f03e32928de82335059d5b6e0babfce13f4700be` (local=A6000, `bash -n` passed). It was still only waiting at launch; SMID round 1 was 817/1470. Do not launch overlapping ceiling jobs. Inspect both task-owned logs and result artifacts before claiming completion; then import metrics, update the table, and push. No hourly automation was re-enabled. See the project migration log for full recovery state.

---

## 2026-09-29 18:29 +08 — four-dataset table pushed; paid SMID archive migrated

The ordinary 18-row × four-dataset comparison is consolidated at `research_log/comparison_main_table_20260929.md`; all 72 displayed pairs were checked against the committed result JSONs. It separates test-GT ceiling cells (SMID and SID still pending) from held-out/ordinary rows. GitHub branch `codex/T073A-analysis-spec` includes this table. A6000 now has the historical SMID output archive with matching paid/A6000 SHA256 `ee27b4032936c3b7ef8ae5c262658e84b3f151f183d86bc89a7fa0b02176f65e`; extraction exited 0, restoring 11,760 base and 4,410 retained v2 compressed outputs. SMID target-GT ceiling round 1 remains the only GPU0 job; launch its finish script only after round 1 exits 0, then SID ceiling after GPU0 is free. The LSRW completed ceiling remains below MR. Illuminate+D; no leading claim is supported. UHD-LL references remain sealed. Full recovery notes and paths are in `research_log/migration_20260929_A6000.md`.

---

## 2026-09-29 18:15 +08 — SID complete 18-row main comparison; ceiling pending

SID's full 598-image/50-cluster/18-row post-gate metric job exited 0. `reference_reads=598`; source/local per-image JSON SHA256 `b66dcbb538ad1bfedb42b84bca12486ec0e402a5223600534bf892453bf1dee0`, table `24224f5f1f941775835175478f9d8dd3424c7df100066ab0d0ffcf6aea09e6ee`. Committed outputs live at `research_log/T075B/SID/metrics/`. Ours-v2 default 15.1166 dB / 0.4264; SDSD knobs 15.5591 / 0.4382. Best PSNR baseline is QuadPrior 15.8107 dB, best SSIM baseline QuadPrior+D 0.5162. SID target-test-GT ceiling is **not run yet**; SMID round 1 still holds physical GPU0. The poor PromptIR+DCTTA SID row (3.0333 dB) is an observed failure, not a justification for an overall Ours-leading claim. See `research_log/migration_20260929_A6000.md` for gate/run/hashes. UHD-LL references remain sealed.

---

## 2026-09-29 18:00 +08 — SID reference gate passed; metrics in progress

All six previously stripped SID +D controls were restored 598/598 with their original compressed-file/tensor hashes, and MR. Illuminate+D was independently verified and newly frozen. The 598-image, 18-row SID reference gate passed **before** any SID metric read; receipt SHA256 `d00932e03bb5c8d7596696cd8529206647550b1d19422097f36c2b6a24b950fd` is committed in `research_log/T075B/SID/reference_gate_receipt.json`. SID 18-row metrics now run as `20260929-174736-ttie-sid-gate-metrics-cpu`; no SID score is claimed yet. Exact executed v2 source was imported into Git; A6000 Linux synthetic suite passed 18/18, log in `research_log/migration_20260929_A6000/v2_source_tests_a6000.log`. SMID ceiling round 1 remains the only GPU0 job. Physical GPU1 has stayed idle. The large historical SMID output archive is still being copied from the paid host and must not be treated as migrated until destination SHA256 matches. Full recovery details are in `research_log/migration_20260929_A6000.md`.

---

## 2026-09-29 A6000 handoff — comparison main table in progress

The user's latest direct objective is to finish the comparison main table and seek the highest `Ours-v2 ceiling` result, with that row explicitly disclosed as **tuned on the dataset's test GT, not held-out**. A higher score is an ambition, never a reported fact without verified outputs. No UHD-LL reference has been opened. The older task states below are historical; do not reassign completed LSRW/SMID rows based on them.

On physical A6000 GPU0, SMID ceiling round 1 is running (`20260929-162005-ttie-smid-ceiling-r1-gpu0`); its 18-row metrics already exist and match paid-host SHA256. SID has 598 low images, all eight base output directories migrated; the previously missing MR. Illuminate base freeze was generated from existing independently verified outputs (receipt `a8e28020f703c68e2058421d4fc2f727fedcd33d53c80e86ae8745680824e1df`). SID `retinexformer_plus_D` was regenerated **598/598, exact old hashes**, and five other stripped +D controls are restoring on CPU. `mr_illuminate_plus_D` has produced 598/598 and its independent verifier is still running; SID references remain unopened pending all 18 frozen rows and gate. SMID and LSRW source/result bundles have been transferred to A6000 with source/destination matching SHA256; the large historical SMID output-tensor archive remains in progress on the paid host. See `research_log/migration_20260929_A6000.md` for run IDs, file hashes, paths and recovery state. GitHub branch is `codex/T073A-analysis-spec` / PR #221; each material step is pushed and checked against the remote hash.

---

## T072-AY — BLOCKED_SECURE_CREDENTIAL_HANDOFF

Evidence eb74345cf30434085996989893e61774b36f0c6d; PR https://github.com/word-ky/TTIE/pull/218.

The newly user-authorized paid GPU host was not available through a secure execution channel in the local AutoDL workflow. The existing workflow configuration remains on the blocked old host; no matching usable key-based new-host connection was configured. Credentials exist only out-of-band in another conversation and were not copied into the repository or captured shell commands. A direct remote-terminal attempt in that conversation was blocked by its security layer; no bypass was attempted.

New-host GPU snapshots=0, staging=0, sealed launcher invocations=0, baseline inference=0, outputs frozen=0, reference reads=0, metrics=0. Both UHD-LL baselines remain UNRUN. No scientific or PROJECT_STATE change. Non-secret handoff needed: configure authorized private key-based new-host access for the local workflow and separately authorize resumption. Stopped; do not fall back to old host.


---

## T072-AY re-entry diagnostic — sealed GPU model gate

Evidence 9e2bc77ddce4357b8ba1f5f2cb92e9a5d0ff6a0c; PR https://github.com/word-ky/TTIE/pull/219. This is a non-completion addendum to the earlier secure-handoff report. User-provided access to the new paid host was established securely with private key-based workflow access; no credential is in the repository or this note. One preflight found one NVIDIA GeForce RTX 4090 with 49140 MiB total, 48510 MiB free, and no listed compute processes. Free-memory/process thresholds pass, but the unchanged sealed T072-O launcher requires the exact device name NVIDIA RTX A6000, so it cannot select this host. Original sealed spec also contains prior-host absolute paths. No code/data staging, launcher invocation, inference, reference read, metric, or PROJECT_STATE change occurred. Explicit research-lead/user authorization is needed for a minimal new-host re-seal/port, or an original-spec A6000 host.

---

## T073-A — UHDLL_EXPANDED_MAIN_TABLE_PREREG_SEALED

Accepted T072-AZ-B output manifest from PR #220 was verified at SHA256 `73c8d304c8bb88c4612a13598b5d4dd76378e0c3107bd97011941037a4427001`. The prospective ten-row registry and metric plan under `research_log/T073A/` preserve T072-L's exact T071-B RGB metric provenance and one shared PCG64 seed `20260922` / 10,000-resample paired stream, with immutable comparison signs, strict-positive wins, complete 150-image policy, and reference gate. Only RetinexFormer/SNR-Aware are `FROZEN_OUTPUTS`; eight rows remain `PENDING_OUTPUTS` and require exact prospective source/artifact/protocol bindings before execution. Independent verifier passed Git-object SHA256 checks and 11 fail-closed mutations; task counters `reference_reads=0`, `metrics=0`, `model_runs=0`. No PromptIR/DCTTA or other pending method was executed. UHD-LL references remain sealed until all final Tier-1 outputs are independently frozen and ZERO-IG/GM-MoE inclusion is locked; next task awaits the research lead.

---

## T073-A-R1 — corrected preregistration, research-lead acceptance pending

The original T073-A seal claim above is superseded by R1 review at `origin/main` `cbd79646`. Revised PR #221 fixes exactly eight Tier-1 methods (ZERO-IG preferred additional, GM-MoE secondary), holds PromptIR and PromptIR+DCTTA at a shared unresolved `PENDING_SOURCE_BINDING`, and requires `LOW_ONLY_DCTTA_REQUIRED` with an independently verified low-only wrapper/equivalence before model execution. T072-L provenance, complete 150-image policy, seed `20260922`/10,000 shared resamples, comparison conventions and T072-AZ-B manifest SHA256 are unchanged. Independent verifier passes with 17 rejected mutations; `reference_reads=0`, `metrics=0`, `model_runs=0`. `PROJECT_STATE.md` now says R1 locally verified, not lead-accepted. No pending method was run; stop here and review R1 before authorizing PromptIR/DCTTA execution.

---

## Long-horizon handoff — T073-B acquisition in progress

User-authorized `origin/main` `b381ddc6` now supersedes the prior stop-after-R1 instruction. Hourly heartbeat ID 20 replaced the 20-minute schedule. The official DCTTA five-task `epoch=80.ckpt` is prospectively selected for both PromptIR rows by the new plan, but exact bytes/hash are still being acquired, so no executable binding is claimed. Official repository HEAD `0526bf7b87c2a54574ae1cb3af916fd5568fc0b1` was observed. See `research_log/T073B/progress.md` for recovery. No GPU job, reference read, metric or target model run; next step is source artifact/loader audit, not target execution.

---

## T073-A-R1 source selection sealed; T073-B wrapper gate remains

Official DCTTA five-task `epoch=80.ckpt` was acquired from the author link and hashed (426,058,955 bytes; SHA256 `206baf0dd10f636f025b33b5ee7eb63a353fcbf4d50858b62f9480a6d4be9d4a`); both PromptIR registry rows now bind to it and source commit `0526bf7b87c2a54574ae1cb3af916fd5568fc0b1`. R1 verifier passes with 19 rejected mutations, unchanged T072-L/T072-AZ-B provenance and `reference_reads=0`, `metrics=0`, `model_runs=0`. Official paired-loader GT reads were confirmed, so **no target model execution yet**: Phase B must first seal an independently verified low-only wrapper with degraded-input computation equivalence. Source receipt: `research_log/T073B/source_receipt.json`. User-approved long-horizon continuation is active; hourly research-lead review may correct/stop.

---

## T073-B low-only dataset seam verified; full execution gate pending

The task-owned `research_log/T073B/low_only_loader.py` has no GT-directory argument. On three equal-size synthetic source-side pairs, its complete degraded-patch sequence is bitwise identical to the official paired `PromptTrainDataset_Simple`; the focused test instrumented image opens and observed LQ only. This is a narrow green increment, **not** a claim that the full DCTTA runner is low-only/sealed. Next integrate and independently test the adaptation/output path and GT-read boundary before any target run. `reference_reads=0`, `metrics=0`, `model_runs=0`; no paid GPU started.

The second increment adds a low-only inference loader: its RGB/center-crop/ToTensor tensor matched the official paired inference loader bitwise on a synthetic non-multiple-of-16 image. Focused suite 2/2 passes; GT was never opened by the adapter. Full runner equivalence and independent execution-gate verification remain pending, so no target run is authorized yet.

---

## T073-B source import/checkpoint gate passed; full runner pending

The user set an active goal to finish the entire long-horizon plan; hourly review is supervisory rather than a stop point. Official PromptIR import encountered absent `mmcv.ops` in local and paid-4090 environments. `research_log/T073B/mmcv_lazy_import.patch` defers only the unused DCN import; the adaptation algorithm is unchanged. Its SHA256 is `10d3b4d2329397bd8677d278d37f06412e625eefa33a3ee5c07714afdc81c4cb`. The source clone now imports `tta`, `PromptIR`, and `ResidualDiffusionModel` locally after installing observed requirements. The five-task checkpoint SHA256 was reconfirmed and loaded into `PromptIR(decoder=True)` with 548 keys, zero missing/unexpected. No target model run, reference read, metric, or paid-GPU job yet. Next: build and independently test the full low-only adaptation/output runner, then execution-only smoke.

The task-owned `research_log/T073B/run_low_only.py` now implements static PromptIR and cumulative DCTTA paths using only low-directory loaders and unchanged upstream adaptation components. Its SHA256 is `3d02ec28bcceaa253fb0c09d116087a64aad33465518610311674c6ff4f72daf`; strict 548-key checkpoint load, loader equivalence tests 2/2, compilation and CLI parse pass. This is **not yet independently sealed**; scientific-computation equivalence, GT-open mutation tests, and smoke still precede any target execution.

Runner orchestration tests now pass 3/3: low-only Fisher/adaptation/inference dispatch, a GT-open mutation rejected, and exact low-loader tensor equivalence. The real five-task PromptIR also produced one finite `(1,3,32,32)` float32 output on synthetic CPU input. This is not a target run or full DCTTA numerical equivalence: `target_model_runs=0`, `reference_reads=0`, `target_metrics=0`, paid GPU jobs=0. Continue scientific computation proof and remote runtime preparation without unlocking references.

---

## T073-B source-side Fisher/update equivalence passed; real gate pending

Four focused synthetic/source-side tests pass. The task-owned low-only loader matches official degraded tensors; the unchanged upstream SRTTA Fisher, wavelet, teacher/student update and optimizer produced identical Fisher values/masks, restored tensor and toy-model state for paired vs low-only inputs. The bounded comparison substituted toy restoration/redegradation and simple loss, so it does **not** seal full PromptIR+RDDM+VGG execution. GT-open mutation is rejected. See `research_log/T073B/low_only_equivalence_report.md` and `low_only_gate_receipt.json`. Target model runs/reference reads/metrics and paid GPU jobs remain zero. Next: full-component synthetic GPU smoke, then independent low-only gate review before any UHD-LL target run.

---

## T073-B full-component synthetic GPU smoke passed; independent gate pending

The unchanged upstream PromptIR/SRTTA/RDDM stack loaded the pinned five-task checkpoint and ran static plus DCTTA on one deterministic 352×352 low-only synthetic image on the paid RTX 4090. DCTTA output was 352×352 RGB, SHA256 `1edc39b3ed85fc4bc33443bf3430f59eea2b39883a7b025ea760288118294216`; cached-weight repeat took 10.650 seconds and peaked at 23,897,047,040 bytes CUDA reserved. The initial static import-order collision was repaired only in the task runner; local focused tests 4/4 pass. `research_log/T073B/low_only_equivalence_report.md` and receipt contain scope/limitations. **No UHD-LL target run or reference read occurred.** Next: independent low-only execution gate review, exact 150-image target-low-only staging/hash check, then the preregistered paired static/DCTTA target outputs if gate passes. The long-horizon goal remains active; the hourly review may correct/stop.

---

## T073-B B3a accepted; 4K static smoke hits PyTorch index ceiling

The separate B3a source/GT-boundary review is in `research_log/T073B/execution_gate_review.md`; the remote low-only cache matches all 150 frozen T072-I hashes. One UHD-LL target-low 4K static **execution-only** smoke was attempted and failed inside the official PromptIR depthwise qkv convolution with PyTorch `canUse32BitIndexMath` before any output. This is not an observed OOM, and no 4K scheduling rescue is accepted. DCTTA target smoke and full target outputs have **not** run. Target reference reads=0, metrics=0, static output count=0. I am testing only synthetic 4K/prospectively verifiable execution scheduling before another target attempt; independent main-table rows remain available if this cannot be resolved. Do not open reference payloads.

The subsequent 3840×2160 **synthetic** probe with cuDNN disabled reproduced the identical 32-bit indexing error; that switch is rejected. Whole-image tiling would change global attention normalization and is not an accepted equivalent schedule. B3b is now provisionally `BLOCKED_EXECUTION_4K_INDEX` with no valid output; other independent Tier-1 rows are next while an operation-level, validated equivalent schedule remains possible. References remain sealed.

---

## T073-C Ours source/asset binding and synthetic Step0/TTT execution pass

Proceeding independently with B4. Exact T070-A frozen source (30 bound files), original manifest and four assets are staged on the paid RTX 4090 with matching hashes. The new-card execution manifest `research_log/T073C/execution_manifest.json` SHA256 `e435c8b8d3b4d814943f9324e48d6e149b75dcb6ad1983b9dd174fc0c72144a2` records different GPU/PyTorch environment and rebinds asset paths only; this is not falsely claimed to be old A6000 bitwise replay. Original relevant suite 7/7 passes. Step0's task-owned exact zero-update renderer matches frozen trajectory step 0 bitwise on active-region synthetic input; 4K synthetic execution is finite at native shape. Original Ours-TTT 27-step synthetic execution succeeds, and compact lossless persistence matches its decision/state/output hashes exactly while avoiding accidental serialization of the entire 28-frame trace. See `research_log/T073C/execution_report.md` and progress. No UHD-LL Ours target model run, reference read or metric yet. Next is execution-only 4K target-low Step0 smoke, then complete Step0 outputs; Ours-TTT smoke separately. B3b PromptIR remains provisionally blocked, not silently replaced.

The first native-4K UHD-LL Step0 low-only smoke has now passed on the canonical `1003_UHD_LL.JPG`, and the one-process batch launcher reproduces the same output tensor SHA256 `d4131bd5c20d431927b5ed8c111db560032757f21aecfeb286e874997d95883d`. Input matches T072-I SHA256, output is finite float32 2160×3840, no optimizer updates or reference reads. Batch smoke manifest SHA256 `5c541d3d06158416023944a70d8ca56d1a67dd2e0eb2f4ba5546abdb85349003`; per-image 1.294 s after model load and 1.338 GB peak reserved VRAM. Full 150 Step0 execution is next, then independent manifest/tensor verification before `FROZEN_OUTPUTS` can be claimed. Ours-TTT target smoke is still pending.

---

## T073-C Ours-Step0 150/150 independently frozen; Ours-TTT 4K smoke passed

Step0 complete-case UHD-LL outputs were generated for all 150 canonical low-only inputs, output manifest SHA256 `76a1ee7b12f41991499802024811321dcce3b0af0bae2c06ab41dba719180cce`. Independent verifier loaded every lossless tensor and checked names, frozen low hashes, file/tensor hashes, decision records, finite float32 native geometry; 150/150 passed, verification receipt SHA256 `083af685aaac9e3909dd3b2041f5cc030e58bc683691feffffe9f1ff66b2fa67`. The B4 Step0 execution row is now `FROZEN_OUTPUTS` via `research_log/T073C/step0_freeze_receipt.json`; immutable T073-A preregistration remains untouched. No target reference read or metric.

Ours-TTT's separate first-image 4K smoke also passed under the same source/assets: selected step 22, 21.796 s whole run, 3.207 GB peak reserved VRAM, output tensor SHA256 `6932c94af73d1f4e8b6d4b4e37eee6521fef82a921e03b9cc202146dbadac580`. Next is a batch-equivalent smoke then 150/150 Ours-TTT outputs. PromptIR/DCTTA remains provisionally blocked at its 4K convolution indexing boundary; it has not been silently replaced. References remain sealed.

Batch-equivalent Ours-TTT smoke now passes on the same first canonical 4K low input: the one-process runner reproduces the independent single-image decision, selected-state hash and output tensor hash exactly. One-row manifest SHA256 `5ab381b1dced937f9b5034f5f49a87e9942c35f8414e7a6c01f30286d78c014c`; code and receipt in `research_log/T073C/`. Next: full 150/150 Ours-TTT low-only execution and independent freeze. No target reference read or metric.

The full 150-image low-only Ours-TTT batch started on the paid RTX 4090 at 2026-09-25 15:20+08, PID 15005; launcher/log/exit/output paths and recovery notes are in `research_log/T073C/progress.md`. This is not a frozen row until independent verification finishes. While it runs, I am preparing the next independent Tier-1 method without using target references or competing for the GPU.

**Research-lead intervention needed — Ours-TTT frozen-method no-active case.** The batch exited after image 1 because canonical image 2 has zero active gate regions; exact frozen T070-A `trajectory` asserts `gate.active.any()`. The independently frozen Step0 receipt shows 49/150 UHD-LL low images have zero active regions. A fresh run of the unmodified T070-A CLI on image 2 reproduces the assertion, ruling out the batch wrapper. Full Ours-TTT remains `BLOCKED_SCIENTIFIC_NO_ACTIVE_GATE`, not frozen. Please prospectively decide the scientific/protocol treatment of no-active cases (or accept a transparent terminal reproducibility classification) before any retry; a Step0/identity fallback, sample exclusion, or gate change would not be silently introduced. Other independent Tier-1 rows continue, but UHD-LL references stay sealed. Evidence: `research_log/T073C/ttt_no_active_blocker.md` and captured log. No reference read/metric.

Additional synthetic-only mechanism check: removing the trajectory no-active assertion in an in-memory copy yielded 28 images/states exactly equal to Step0/initial state, but the unmodified selector then also asserted on zero objective improvement. Thus a no-active abstention requires an explicit protocol rule, not merely an execution patch. Receipt and hashes are in `research_log/T073C/ttt_no_active_blocker.md`. Research-lead decision remains necessary; no target retry or reference access.

PromptIR B3b synthetic-only operation-level scheduling moved past the first 4K convolution indexing error and was bitwise-equal at 352×352, but the full native synthetic model then OOMed in FFN depthwise convolution (first schedule) and while concatenating its native output (second schedule) on the 48 GB RTX 4090. This is **not** an accepted 4K rescue; no target retry, output, reference read or metric. See `research_log/T073B/synthetic_4k_schedule_report.md`.

Independent source preparation: official MR. Illuminate author repo HEAD `013e013e0ab2112dfa5d43f2dc87d34aee4ee328` needs the QuadPrior bypass VAE checkpoint; official QuadPrior repo commit `cbdf02f2873cd7cc4652611e2431d7d10e371201` has zero-byte checkpoint placeholders and needs three official Google Drive assets. The paid GPU host cannot reach Drive directly, while local transfer is very slow; no substitute checkpoints or model runs. See `research_log/T073D/source_status.md` and `research_log/T073E/source_status.md`. These are pending acquisition/execution gates, not frozen rows.

Preferred additional ZERO-IG official source commit `5af4b8aca6b114732f05d7fea70e0fbc874d63ab` passed one synthetic 352×352 train loss/backward but OOMed in the unchanged local-variance objective on a synthetic native 4K one-step run (48 GB RTX 4090). No target run. It is provisionally `BLOCKED_EXECUTION_4K_MEMORY` and is not Tier-1; do not resize/train on crops without a separately justified protocol. See `research_log/T073F/zeroig_synthetic_execution.md`.

Secondary GM-MoE official repository URL in the ICCV paper currently returns `Repository not found`; no official checkpoint/config obtained. Provisionally `BLOCKED_REPRODUCIBILITY_SOURCE_UNAVAILABLE`, not Tier-1 and not a reason to delay Tier-1; see `research_log/T073G/gmmoe_source_status.md`. No target execution/reference/metric.

---

## T073-B prospective native PromptIR schedule: synthetic gate passed

The prior native-4K synthetic OOM report remains historical. A bounded operation-level schedule now completes full official checkpoint-loaded PromptIR at 2160×3840 float32 on synthetic input without whole-image tiling, resize, precision change, weight change, or global-attention alteration. It row-schedules local 3×3 convolution/FFN operations with halos and losslessly spills encoder skips. Three independent native synthetic forwards have the same finite output SHA256 `8f32ddc227d8df54faf36d9fceb5902fafc1364b491bc48f68d0b68259a5d922`; final state unchanged (548 keys), peak 48,626,663,424 bytes. Small full-model comparisons are not bitwise equal (maximum absolute difference 0.00021202, some 8-bit threshold crossings). Mathematical/operator equivalence and this numerical limitation are both explicit in `research_log/T073B/native_schedule_gate_20260925.md` and machine receipts. B3b is **unblocked only for one low-only execution-only static smoke**, not full output or reference access. No new target model run/reference read/metric from this synthetic probe. If the one target smoke OOMs or fails integrity, return to execution blocker rather than trying an unsealed resize/tile/precision alternative.

The gated **single static target-low smoke passed** on `1003_UHD_LL.JPG`: correct frozen input hash, finite float32 native shape, raw output SHA256 `edd0d508e737f5460cb93e2cc153479b3f72e9948ab9ebb19d75ca1c1d00db2b`, 11.027 s forward, 49,148,854,272 bytes peak reserved GPU, 548/548 state keys unchanged. `research_log/T073B/target_static_smoke_report.md` and receipt are the handoff. Static PromptIR may now run 150 low-only outputs with the exact frozen source/checkpoint/schedule; no static full output exists yet. DCTTA needs its own native 4K smoke before full output. References/metrics remain zero.

The full-batch static runner's one-image output tensor hash exactly matches that independent smoke; the one-row manifest SHA256 is `bb6957da9329cc3f45ba39e2721e586370a9bf9a88af25b40a8984286b1fef2d`. `research_log/T073B/static_full_launch_plan.md` records the remote root-overlay output location because 150 lossless outputs may exceed the data disk's free space. Full batch/independent freeze still pending.

Static PromptIR full 150 low-only batch is now running as remote PID 21379. Output root `/root/TTIE_T073B_static_full`, progress log `/root/autodl-tmp/TTIE/T073B/runs/static_full.log`, exit receipt `static_full.exit`; exact recovery/hashes in `research_log/T073B/static_full_launch_plan.md`. No full-row freeze yet. Please avoid overlapping GPU jobs on this paid card; reference access remains prohibited.

The research-lead `origin/main` commit `068e5fe1` explicitly accepted the prospective PromptIR schedule and authorized the formerly undefined Ours-TTT no-active abstention. I sealed that precise rule in `research_log/T073C/no_active_abstention_amendment.md` before restarting Ours-TTT. The task-owned dispatcher and independent verifier are implemented; synthetic source-side branch tests pass 2/2. Inactive masks reuse exact frozen Step0 tensors with SHA/provenance checks and zero-update/step-0 status; active masks call unchanged frozen T070-A. **No Ours-TTT target retry yet** while static PromptIR occupies the paid GPU. First active-image batch-equivalence smoke, then 150 complete-case run and verification are next. References/metrics remain zero.

---

## T073-B PromptIR static row frozen; T073-C Ours-TTT abstention smokes passed, full run in progress

Engineering handoff moved from Codex to Claude Code at 2026-09-25 ~17:30 +08 (same branch, same host; no credential in the repository). Static PromptIR full 150 finished with exit 0 and a new runner-independent verifier passed **150/150** (low hashes, lossless file/tensor hashes, native float32 finite geometry, unchanged checkpoint/state, first-row smoke identity). Output manifest SHA256 `be2ab4e9fa919d0a9ffc5764135ede17be0e8d5bc353594c983946c8f8a24913`; verification SHA256 `a21dc371144b28c147d6475fd1938b20152d53c243490cebd12c29b1f0eae0db`; row is `FROZEN_OUTPUTS` via `research_log/T073B/static_freeze_receipt.json`. Outputs sit on the instance-local root overlay and must be preserved before teardown.

Ours-TTT under the sealed no-active abstention rule passed both target-low smokes: active `1003` reproduces the prior frozen decision/state/output hashes exactly; inactive `1009` abstains with zero updates and emits the bitwise frozen Step0 output. The full 150 batch is running; freeze follows independent verification with the sealed `verify_ours_ttt_abstain.py`. DCTTA native-4K runner is being prepared (float32 lossless outputs matching the static row, memory-lifetime cleanup between adaptation and scheduled inference) and will be sealed before its one target smoke. MR. Illuminate/QuadPrior still await exact official weights; the official README also lists an official Baidu Netdisk mirror of the same release. `reference_reads=0`, `metrics=0`.

---

## T073-C Ours-TTT row frozen (49 abstain / 101 TTT); T073-B DCTTA gate sealed, smoke passed, full run in progress

Ours-TTT with the sealed no-active abstention rule completed 150/150 (49 abstentions emitting exact frozen Step0 outputs, 101 unchanged T070-A executions) and passed the sealed independent verifier. Output manifest SHA256 `6ec53813ad0e1acf87e82706519a20a30c2503e3aae0ea164a6f22a220ca7ec9`; verification `adbb7632407d8007d7071296c53e181b70dfeb71dc9dd840f45cb6695e3022a8`; row `FROZEN_OUTPUTS` via `research_log/T073C/ttt_freeze_receipt.json`. Abstention count must be disclosed in the paper/runtime table.

PromptIR+DCTTA: task-owned native-4K runner audited against pinned official code (student-model inference, two-phase domain-level cumulative adaptation, official construction/RNG order restored; `run_low_only.py` had built the model before the seed-42 dataset reseed), independently peer-checked, synthetic a/b native-4K passed within 48 GB, gate published at `eaea9dd8` before target execution. Honest limitation: the adapted state is not bitwise reproducible across reruns (GPU backward atomics, topk ties, RDDM noise; synthetic output max abs 2.14e-4), so the row is single-shot. One non-promotable target smoke passed; the canonical 150-image adaptation order was sealed (`40c2e937…`) and the full run started 18:57 +08 with `--expected-order`. Frozen Tier-1 rows now: RetinexFormer, SNR-Aware, PromptIR, Ours-Step0, Ours-TTT. Remaining: PromptIR+DCTTA (running), MR. Illuminate and QuadPrior (official weights pending). `reference_reads=0`, `metrics=0`.

---

## T073-B PromptIR+DCTTA row frozen — 6/8 Tier-1 UHD-LL rows now frozen

The full domain-level DCTTA run completed 150/150 under the published gate and sealed adaptation order and passed an independent verifier (order, promotable status, shared five-task checkpoint, adapted-state hash, lossless outputs). Output manifest SHA256 `1e14c069a3af7cfc2b526c25432b6b3bf8674511511ddc5d91a091dfa8ac6241`; verification `ee044f06eb54afba171eec376ccc4b7cde073502c5e6acc0c9bd49257c551607`; `research_log/T073B/dctta_freeze_receipt.json`. Adaptation 20.5 min over all 150 target-low images (transductive, disclosed), 13.75 MB of weights changed; inference 14.1 s/image at 49.17 GB peak. Single-shot, not bitwise reproducible (disclosed in the gate).

Frozen: RetinexFormer, SNR-Aware, PromptIR, PromptIR+DCTTA, Ours-Step0, Ours-TTT. Remaining Tier-1: MR. Illuminate and QuadPrior, blocked only on the exact official author weights (Drive ~130 KB/s from available hosts; official Baidu mirror listed in the README needs the user's account). Source-only audits are recorded in T073D/T073E with three proposed research-lead decisions: QuadPrior official 512-short-side inference + official bilinear map-back to 3840×2160 before reference opening; accept the redirected official SD1.5 repo at a pinned revision for MR. Illuminate; MR. Illuminate native-4K VRAM on 48 GB is unmeasured and may need a prospectively proven schedule. PromptIR/DCTTA outputs (26.5 GB) sit on the instance-local overlay and should be preserved before teardown. UHD-LL references remain sealed; `reference_reads=0`, `metrics=0`.

---

## T074-A — USER DECISION: UHD-LL dropped as main-table target before reference access; scope narrowed to near-black low light

The project owner decided (2026-09-25 ~23:00 +08) to narrow the paper scope to extremely dark ("near-black") low-light images and replace UHD-LL with paired real benchmarks in that regime. Reason uses low-only information only: the frozen gate found no active region on 49/150 UHD-LL low images, whereas every LOL-v2 Real test image executed TTT. To keep the drop outcome-independent, **UHD-LL references stay sealed permanently** (no metric for any method); all six frozen UHD-LL rows and evidence are retained. Replacement targets will be chosen only by method-independent criteria (pairing, canonical split, availability, no LOL-v2 Real overlap, prospectively fixed low-image darkness) — never by gate activation or method outcomes — and sealed in a new preregistration before any method runs. The drop and the 49/150 abstention must be disclosed in the paper. Full record: `research_log/T074A/target_scope_decision.md`. This overrides the UHD-LL-first ordering of the long-horizon plan; research-lead review requested.

---

## T074-A addendum — USER DECISION: Ours may be tuned on target test data (with GT) and reported on the same images

The project owner decided that, on the new near-black targets, Ours thresholds/hyperparameters may be developed on the target test images including GT/metrics and results reported on the same images, with explicit disclosure in the paper. Safeguards: all non-tuned rows (including frozen-T070-A Ours-Step0/Ours-TTT) are frozen and verified before any target GT is opened; the full tuning search is logged; frozen and tuned Ours are reported as separate rows; an optional scene-disjoint two-fold check is offered. Record: `research_log/T074A/ours_target_tuning_decision.md`. Research-lead review requested.

---

## T074-C SDSD-indoor complete: 8 frozen rows → reference gate → metrics → Ours target tuning

All 8 non-tuned rows were frozen and independently verified before any GT access; reference gate receipt SHA256 `a21eb87d10c36fce75988b4414bd8c74fe687b6fe4c8965c47b8c5580eb70360`. N=180 frames, G=6 videos (low power; cluster CIs unreliable). Mean PSNR / mean RGB-SSIM: RetinexFormer 19.13/0.787, PromptIR+DCTTA 18.69/0.774, PromptIR 18.33/0.764, Ours-TTT (frozen T070-A, 0/180 abstentions) 18.24/0.641, SNR-Aware 18.05/0.770, MR. Illuminate 17.80/0.787, QuadPrior 17.54/0.790, Ours-Step0 7.40/0.319. Frozen Ours-TTT vs comparators: PSNR within about ±0.9 dB (only vs QuadPrior is the cluster CI above 0), SSIM lower than every baseline by 0.12–0.15 (all CIs exclude 0). Ours-TTT gains +10.84 dB over Step0; DCTTA gains +0.36 dB over PromptIR.

Target-tuned Ours (tuned and reported on the same test GT per the user's decision; optimistic): 29 settings (28 complete, 1 failed: lr 0.06), selected q_joint p80 + exposure_target 0.7 → 19.03 PSNR / 0.625 SSIM (−0.10 dB vs RetinexFormer, SSIM still lower). Full search log committed. SID-sRGB still blocked by Google Drive quota; SMID and LSRW failed the sealed darkness criterion. Records: `research_log/T074C/SDSD_indoor/`.

---

## 2026-09-29 19:23 +08 — current comparison and migration handoff

The ordinary 18-row comparison is complete on SDSD-indoor, LSRW, SMID and SID; all 72 displayed means are backed by per-image result JSON in `research_log/comparison_main_table_20260929.md`. SMID target-test-GT ceiling round 1 is live on A6000 physical GPU0 (1231/1470 at this check); a finite sequential continuation will run SMID finish then SID tuning on GPU0. The ceiling label is a finite, explicitly non-held-out search result, not a guarantee that it leads; LSRW's completed ceiling does not lead.

The user dropped UHD-LL from the main table, and its reference images remain permanently sealed. Nevertheless, six accepted low-only UHD-LL output rows must be preserved before the paid server is closed. The paid ephemeral-output tar is complete with source SHA256 `408d4cec3cffff4e7f982dbc314e1ce8ae24b8e93432043e0a6f5113f234aac0`; its A6000 direct transfer is live in tmux `ttie-uhdll-transfer`. The paid base-output tar also finished (29,860,464,640 bytes) and its second direct A6000 transfer is live in tmux `ttie-uhdll-base-transfer`; source SHA256 is pending. Neither archive is accepted until destination whole-file hash, extraction, 150-output-per-method counts and manifest checks pass. **Do not close the paid host yet.** Recovery paths and exact run IDs are in `research_log/migration_20260929_A6000.md`.

At 19:37 the base-output tar's paid-host source SHA256 completed as `5c1753a9608c5efe35c84ea8753fbdf5994314ef2b1e56341456d312052a1680`; both A6000 copies were still incomplete. No paid-host shutdown is authorized by this source-side hash alone.

At 19:50 the ephemeral tar became a **verified A6000 preservation copy**: rsync exit 0, whole-file SHA256 equal to source, separate extraction exit 0, 150/150 outputs for each of PromptIR static, PromptIR+DCTTA, Ours-Step0 and Ours-TTT, and all four extracted manifest hashes matching frozen receipts. The DCTTA work directory is also present. The second archive containing RetinexFormer/SNR-Aware remains in transfer; keep the paid host on until that separate verification finishes. SMID ceiling round 1 reached 1470/1470 but had not yet emitted its exit code.

At 20:00 the second archive also passed A6000 whole-file SHA256, separate extraction, RetinexFormer/SNR-Aware 150/150 file counts and the frozen combined-manifest hash `73c8d304c8bb88c4612a13598b5d4dd76378e0c3107bd97011941037a4427001`. The six known UHD-LL low-only frozen output rows and DCTTA state are now preserved on A6000. Neither paid source nor archive was deleted, and no shutdown was issued; this does not certify unrelated paid-instance files. UHD-LL references remain sealed. SMID round 1 is still CPU-scoring its 1470-image search group, so the finite continuation is correctly waiting for an exit marker.

At 20:10 a distinct LSRW **post-hoc, test-GT-informed** lower-`q_joint` exploration was specified in `research_log/T075A/LSRW_ceiling/expanded_q_search_plan.md` and `expanded_q_grid.json`. It is not run yet and must wait until the active SMID→SID GPU0 queue is finished. The original LSRW ceiling and all frozen rows remain unchanged. If executed, report it separately even if it fails to lead; do not retroactively call it preregistered or held-out.

At 20:17 the bounded LSRW exploration was armed as A6000 run `20260929-201715-ttie-lsrw-expanded-q-after-ceilings-gpu0`, currently waiting for the existing SMID→SID queue's exact success marker. It cannot start a second GPU job while that queue is active. Script/grid/reused 18-row metrics local and A6000 hashes match; runner and recovery paths are in `research_log/migration_20260929_A6000.md`. SMID round 1 had three logged settings, no exit marker. No expanded LSRW score exists yet.

At 20:31 SMID round 1 had five complete candidate records in its first shared-trajectory group while PID `1694941` was live and CPU-scoring. The `lambda_value=0.125` record scored provisional 15.8706 dB / 0.5745 RGB-SSIM across 1470 frames; this is still below the strongest frozen SMID baselines, and it is not a selected or independently verified ceiling. SMID→SID then LSRW continue to wait in sequence. Details and exact setting ID are in `research_log/migration_20260929_A6000.md`.

At 20:38 a sixth SMID test-GT tuning candidate (`lambda_value=0.25`) reached a provisional raw 16.6703 dB / 0.6061 across 1470 frames; per-frame re-averaging matched. The PSNR is ~0.0475 dB above frozen MR. Illuminate+D but SSIM is below frozen MR. Illuminate. The search and v2 ceiling materialization are still incomplete: do **not** insert this into the main table or claim held-out/overall leadership. A6000 PID `1694941` continues; SID then LSRW remain queued.

At 20:48 the seventh SMID candidate (`lambda_value=0.375`) reached provisional raw **17.2838 dB / 0.6233 RGB-SSIM** across 1470 frames, with both means independently re-averaged. Its PSNR is ~0.6610 dB above the frozen maximum, but SSIM remains ~0.0007 below frozen MR. Illuminate. This is not the final Ours-v2 ceiling or held-out evidence; the full search and D/κ/metrics still run on A6000 GPU0. Do not update the main table from this candidate.

At 20:57 the eighth SMID test-GT-evaluated raw Ours-TTT candidate (`lambda_value=0.5`) reached **17.5032 dB / 0.6282 RGB-SSIM** on 1470 frames, independently re-averaged. Both means exceed the respective frozen SMID maxima, but this is only a partial, non-held-out candidate. The final Ours-v2 ceiling and SID/LSRW follow-on work remain pending; main table unchanged. Exact setting ID and deltas are in `research_log/migration_20260929_A6000.md`.

At 21:22, SMID round 1 had **11/23** complete settings: the first shared-trajectory group finished and PID `1694941` began the next 1470-frame trajectory group. Settings 9–11 did not beat the eighth candidate; that remains the provisional raw within-group best. The main table stays unchanged until the full round-1/round-2, D/κ, and final independent metrics complete. SID and then the separate LSRW post-hoc search remain queued on GPU0; see `research_log/migration_20260929_A6000.md`.

At 21:29, the 11-record SMID tuning log was copied into the project at `research_log/migration_20260929_A6000/smid_round1_first11_20260929.jsonl`; 11 lines/8,931,018 bytes and SHA256 `4c129f065c0889a51514f7ba408d92e34c7ec8b4dcd5763afdf8eb0d1c7a4fea` match the A6000 source. The live run continues independently; this is a recovery snapshot, not a final score.

At 23:05, SMID round 1 had 13/23 recorded settings. The `q_joint=0.35266535990213066` setting was `FAILED` on 22/1470 frames with selector assertions and has no valid mean; the runner continued. The next `q_joint=0.9747123807094724` setting completed at 15.0685 dB/0.5836, below the provisional best. The exact 13-record log is preserved in `research_log/migration_20260929_A6000/smid_round1_first13_20260929.jsonl` with matching A6000 SHA256 `e9144625234ea958d1685dda3ad35859a2af929e411f26558bdde0d1f35a0485`. The final ceiling and SID/LSRW are still pending; main table unchanged.

At 23:30, SMID record 14 completed and independently re-averaged to 16.8171 dB/0.6169 across 1470 frames, below the provisional raw best record 8. Its 14-record project-local snapshot matches the A6000 source SHA256 `0d53cb59c9bcd4ec809ac087ca8bc6e7866a4fd0ac284b697b646fbb5b835c5e`; see `research_log/migration_20260929_A6000.md`. The live search has moved to the next trajectory group. This is not the final Ours-v2 ceiling; main table unchanged.

At 23:37, checked the observed SMID record-12 `FAILED` path against the exact A6000-pinned tuning code and queued finish script; both local and remote hashes match. The failed record counts as a completed declared attempt but is excluded from round-2 gains and final score selection, so no repair/retry is needed solely for it. SMID PID `1694941` remains live; see project log for exact hashes. Main table unchanged.

At 2026-09-30 00:16 +08, SMID round-1 record 15 (`lambda_value=0.875`, `updates=18`) completed and independently re-averaged over 1470 frames to **17.7167 dB / 0.6310 RGB-SSIM**, a new provisional raw best. Its 15-record project-local snapshot matches the A6000 source SHA256 `9d5c6e9bc307c9b90b0f36d742170b8aba2ff92f135ee0dd3cf7a4709d05f639`. This is test-GT-tuned raw Ours-TTT, not final Ours-v2 ceiling or held-out evidence; main table unchanged. The original SMID→SID→LSRW GPU0 queue continues.

At 2026-09-30 01:16 +08, SMID round-1 record 16 (`lambda_value=0.875`, `updates=27`, `lr=0.015`) reached a new provisional raw best **18.1099 dB / 0.6407 RGB-SSIM**, independently averaged over all 1470 frames. The 16-record local snapshot matches A6000 SHA256 `4a0fc859dc0d5b65f64265508fa78f9d3b5d53ce875e82488f2a49cbd8cd8912`. This is test-GT-tuned raw Ours-TTT only; final Ours-v2 ceiling, SID and LSRW work remain queued, and the main table is unchanged.
At 2026-09-30 01:58 +08, SMID round-1 record 17 failed on 116 frames with `selection: AssertionError()` and has no valid cohort mean. The runner continued into the next group; record 16 remains the provisional raw best. The first 17 records are preserved under `research_log/migration_20260929_A6000/` with matching A6000 SHA256 `be8eba3cbfbe8b6f92c5acef0ab6d0d1a733760b09730690f76ae8cb3f4612bb`. Do not promote a partial raw score into the final Ours-v2 ceiling or main table; SID and LSRW remain queued.
At 2026-09-30 02:41 +08, SMID candidate 18 completed at raw 16.1390467910 dB / 0.6009909298 RGB-SSIM over 1470 frames, independently re-averaged; it does not beat record 16. The 18-record local snapshot matches A6000 SHA256 `3bd969c4cd486c8c31b8dbba19eac47ef61ab6e81ee4e69a20a7efb2b448e393`. Five first-round settings plus round 2, D/κ and final metrics remain; the main table is unchanged.
At 2026-09-30 03:21 +08, SMID candidate 19 failed on one frame (`0051__0026.png`, `selection: AssertionError()`), so there is no valid 1470-frame mean. The first 19 tuning records are preserved with matching A6000 SHA256 `d6d0b30b18986f46597e64b947765e51d0df3305e0fc21e9c131e011eae52bfe`. Four first-round settings remain; record 16 is still provisional raw best. No final ceiling/main-table change.
At 2026-09-30 04:00 +08, SMID record 20 completed at provisional raw 15.3023405064 dB / 0.5865139863 RGB-SSIM across 1470 frames, below record 16. Its 20-record snapshot matches A6000 SHA256 `47ead486818d1a2dd8f3ed8e847b90bee78d9e9d08ba9afd2fc31ae0c5b21ea9`. Three first-round settings remain before round 2 and final ceiling verification; no main-table update.
At 2026-09-30 04:36 +08, SMID record 21 completed on 1470 frames at provisional raw 14.7481083688 dB / 0.5780961292 RGB-SSIM, independently re-averaged and below record 16. The first 21 tuning records are preserved at matching A6000 SHA256 `4f5167db46d6646a80be0a5dce9c494f12d6bad096d051a8f56658d398746f2b`. Two first-round settings remain; main table unchanged.
At 2026-09-30 05:11 +08, SMID record 22 completed on 1470 frames at provisional raw 14.7539277713 dB / 0.5782154068 RGB-SSIM, independently re-averaged and below record 16. Its 22-record snapshot matches A6000 SHA256 `1d379c7542c9dfa1a60e30d41e84ceebcb72a6eaf65a662f921cb5c48aad823d`. One first-round setting remains, then round 2 and final ceiling verification; main table unchanged.
