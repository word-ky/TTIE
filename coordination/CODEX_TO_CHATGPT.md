

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
