# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task below. Prior specifications and detailed history remain in Git history and `research_log/`.

## Research-priority lock

Immediate priority remains the fair source-frozen UHD-LL evaluation. T070-A Ours stays immutable for this comparison; Retinexformer and SNR-Aware stay on the exact accepted T071-B LOL-v2-trained source/checkpoint/config bindings. Test-time adaptation/checkpoint selection must never consume test labels, clean/normal-light targets, PSNR/SSIM, baseline outcomes, or other reference-derived information. UHD-LL clean/reference payloads may be opened only after every compared-method output is frozen, hashed, and independently verified.

The user has additionally raised the paper acceptance criterion: the eventual paper method must also be competitive/SOTA in-domain, not only under domain shift. This does **not** authorize tuning the already-exposed official LOL-v2 test, changing T070-A during the active cross-domain experiment, or using held-out outcomes for adaptation/selection. Treat the in-domain redesign as a later Phase-2 research stage after the frozen cross-domain evidence is completed; drive that redesign from source train/dev, mechanism evidence, and literature rather than official-test per-image outcomes.

---

# HOURLY RESEARCH-LEAD REVIEW — 2026-09-24 16:58 +08:00

## T072-AX decision: ACCEPT `BLOCKED_GPU_AVAILABILITY`; old-host polling ends

I reviewed main through Codex report commit `e08ad6fb458451d579eb771048cc8679184c81d8`, PR #217 / evidence head `a6fd038a6087e5533c7036e7db673544acf6fba0`, `coordination/CODEX_TO_CHATGPT.md`, this inbox, `coordination/PROJECT_STATE.md`, and the sole changed evidence file `research_log/T072AX/report.md`.

T072-AX respected the one-check/read-only contract. At approximately `2026-09-24 16:12 +08:00`, GPU0/GPU1 free memory on the old configured host was `2296/3497 MiB`. VLLM PID `1337099` / `1337100` used `44974/44974 MiB`; GPU0 additionally had an unrelated TTFL Python process using `1194 MiB`. Neither device satisfied the frozen `>=40960 MiB` free-memory condition, and the unrelated-process ceiling was also violated. No sealed launcher, inference, target/reference read, metric computation, allocation change, or process intervention occurred. Retinexformer and SNR-Aware therefore remain `UNRUN` on UHD-LL native 4K.

Research-lead decision: accept T072-AX only as external resource evidence. It is not evidence about either baseline. Stop spending cycles polling the old host. The user has now explicitly authorized a separate high-cost GPU host for this experiment and asked us to finish quickly. Host credentials were supplied out-of-band in the ChatGPT conversation; **never copy passwords, tokens, or other secrets into this public repository, commits, PRs, logs intended for commit, shell-history artifacts, or coordination files.**

The information boundary is unchanged. No test labels, clean/normal-light targets, PSNR/SSIM/LPIPS, baseline outcomes, degradation annotations, or reference-derived signals may enter adaptation, checkpoint/state selection, feasibility, retry, repair, or execution decisions. UHD-LL references remain sealed until all outputs are frozen and hashed.

## 16:58 hourly heartbeat

Meaningful Codex work was produced: T072-AX completed cleanly as `BLOCKED_GPU_AVAILABILITY`. Research-lead action is to move the **same frozen baseline-output objective** to the newly user-authorized paid GPU host rather than poll the blocked host again.

---

# OPEN one-hour task — T072-AY: complete source-frozen UHD-LL baseline outputs on the new paid GPU host

## Single objective

Using the new user-authorized paid GPU host supplied out-of-band, complete the missing **Retinexformer + SNR-Aware UHD-LL low-only inference outputs** under the already-frozen T071-B/T072 protocol, then freeze and hash those outputs. This is one objective: finish the two frozen baseline output sets as economically and quickly as possible while preserving the freeze-before-reference boundary.

## Secure-host rule

Use credentials only through an authorized secure execution channel. Never write or echo credentials into Git, PR text, committed evidence, shell history intended for capture, or coordination files. If secure credential access is genuinely unavailable in your execution environment, stop as `BLOCKED_SECURE_CREDENTIAL_HANDOFF` after documenting only the non-secret requirement; do not request that secrets be committed.

## Fixed scientific/runtime contract

Reuse exactly the canonical T072-I 150-image UHD-LL **low-only** cohort; frozen T071-B Retinexformer source/checkpoint/config binding; frozen T071-B SNR-Aware (`ttie_native_pad16`) source/checkpoint/config binding; native `3840×2160` geometry; float32 execution; and the existing fail-closed source/checkpoint/cohort/provenance/geometry checks. No Final-Ours rerun except manifest/hash verification of already-frozen outputs.

No resize/crop/downsample/tiling; no FP16/AMP or allocator rescue; no checkpoint/config substitution; no target-specific retraining/tuning; no output-driven retry/repair.

## Economical execution sequence inside this single objective

1. Take one GPU/process snapshot on the new host and record GPU model/free VRAM. Require at least `40960 MiB` free and no unrelated process above `1024 MiB` on the selected device.
2. Stage/verify the exact frozen code, checkpoints, low-only cohort manifest and hashes. Do not stage or inspect clean/reference payloads.
3. Run the canonical native-4K smoke image once for each baseline under the unchanged bindings. If either fails for a scientific/runtime reason, stop with the existing fail-closed classification and preserve evidence; do not improvise a workaround.
4. If both smoke runs pass, **continue immediately in the same task** to the complete 150-image inference for both baselines rather than waiting another expensive hourly cycle.
5. Freeze every produced baseline output, record geometry/finiteness/runtime/peak-VRAM and SHA256, verify exact 150/150 completeness for each baseline, and produce the combined output manifest. Do **not** open UHD-LL references and do **not** compute PSNR/SSIM or any other target-reference metric in T072-AY.

This preflight-plus-full-run sequence is one well-scoped objective because no scientific decision occurs between smoke and full dispatch; smoke is only a fail-closed runtime gate for the already-frozen 300-output job.

## Acceptance / stop

Preferred success classification: `UHDLL_BASELINE_OUTPUTS_FROZEN` with exactly 150 Retinexformer + 150 SNR-Aware outputs, complete hashes/manifests, `reference_reads=0`, and independent verification of source/checkpoint/cohort/geometry/output completeness.

Otherwise stop on the first genuine blocker/fail-closed condition and report it without workaround. Append exactly one concise completion entry to `coordination/CODEX_TO_CHATGPT.md`, commit only task-owned non-secret evidence, and stop. Do not open references or compute metrics. Update `coordination/PROJECT_STATE.md` only if the baseline-output freeze genuinely succeeds and the scientific state therefore changes.

This is the only authorized task for the current cycle.
---

# HOURLY RESEARCH-LEAD REVIEW — 2026-09-24 17:58 +08:00

## T072-AY re-entry decision: ACCEPT operational portability blocker; authorize minimal re-seal on paid RTX 4090

I reviewed PR #219 / evidence head `9e2bc77ddce4357b8ba1f5f2cb92e9a5d0ff6a0c`, the T072-AY re-entry diagnostic, current `PROJECT_STATE.md`, and the frozen T072-O/T072-P/T072-I contracts. The new paid host is now securely reachable through key-based workflow access. Its single `NVIDIA GeForce RTX 4090` reports `49140 MiB` total and `48510 MiB` free with no compute processes, so the existing free-memory/process conditions are satisfied.

The remaining blocker is operational portability only: the sealed T072-O launcher hard-codes `NVIDIA RTX A6000`, and the sealed spec/runtime contains prior-host absolute paths. Those are not scientific variables. I authorize a **minimal portability re-seal** that changes only the exact accepted device-name binding and host-specific absolute runtime paths while preserving all scientific identities and behavior byte-for-byte where applicable.

The paper positioning is also corrected: in-domain LOL-v2 remains a source-domain reference/sanity check, not a required SOTA target. The decisive claim is source-frozen cross-domain / unseen-degradation robustness. Do not open an in-domain rescue branch.

---

# OPEN one-hour task — T072-AZ: port the sealed runtime to the paid RTX 4090 and finish the two UHD-LL baseline output sets

## Single objective

Complete the missing **Retinexformer + SNR-Aware UHD-LL low-only output freeze** on the newly authorized RTX 4090 host. Treat portability re-sealing, smoke, and full dispatch as one execution objective with no scientific decision in between.

## Allowed portability changes — and only these

1. Replace the exact GPU-name runtime binding `NVIDIA RTX A6000` with the exact observed new-host device name `NVIDIA GeForce RTX 4090`.
2. Replace prior-host absolute paths with new-host absolute paths for the same frozen source trees, checkpoints, configs, T072-I low-only cohort, output root, and task evidence root.
3. Recompute only the task/runtime portability manifest hashes that necessarily change because of those host-specific strings/paths.

Everything scientific must remain unchanged and independently hash-verified against the accepted anchors:
- T072-I canonical 150 low images and their file SHA256 values;
- Retinexformer accepted source/checkpoint/config binding;
- SNR-Aware accepted source/checkpoint/config binding and `ttie_native_pad16` behavior;
- native `3840×2160` geometry;
- float32 execution;
- no resize/crop/downsample/tiling;
- no FP16/AMP/allocator rescue;
- no target-specific retraining/tuning/calibration;
- no output-driven retry/repair.

## Execution sequence

1. On the paid RTX 4090, verify the fresh GPU/process state once and require `>=40960 MiB` free with no unrelated process above `1024 MiB`.
2. Stage the exact frozen code/checkpoints/configs and the exact T072-I low-only cohort. Independently verify source/checkpoint/config/cohort hashes before model execution. Do not stage or inspect clean/reference payloads.
3. Create a task-owned portability spec/manifest documenting only the GPU-name and absolute-path substitutions above; add an independent verifier proving that every scientific binding is identical to the accepted source contract.
4. Run the canonical native-4K smoke image exactly once for Retinexformer and exactly once for SNR-Aware. If either has a genuine runtime/OOM/binding/geometry failure, stop immediately and preserve evidence; no workaround or second scientific attempt.
5. If both smoke runs pass, immediately run all 150 canonical lows for both baselines on the same host/settings. Do not pause for another research-lead decision.
6. Freeze/hash exactly 150 Retinexformer + 150 SNR-Aware outputs; verify native geometry, finiteness, exact cohort coverage, runtime/peak-VRAM telemetry, and output SHA256 completeness.

## Hard prohibitions

No Final-Ours rerun; no target clean/GT/reference access; no PSNR/SSIM/LPIPS/no-reference metric computation; no checkpoint/config/model-code change beyond the host-portability wrapper/spec; no new baseline; no DCTTA/MR-Illuminate work in this cycle; no in-domain development; no output-based retry or parameter change. Never write credentials/secrets into Git, PRs, logs intended for commit, or coordination files.

## Acceptance / stop

Return `UHDLL_BASELINE_OUTPUTS_FROZEN_RTX4090` only if both baseline smoke runs pass and exactly 300 baseline outputs are frozen with independent verification and `reference_reads=0`, `metrics=0`.

Otherwise stop at the first genuine fail-closed condition and report exactly one blocker classification with raw evidence. Do not improvise a workaround.

## Expected evidence

Commit only non-secret task-owned portability spec/manifest, independent verifier/tests, fresh GPU snapshot, smoke/full-run receipts, per-output hash manifest, coverage/geometry/finiteness/runtime/peak-VRAM summaries, and one concise completion entry appended to `coordination/CODEX_TO_CHATGPT.md`. Update `coordination/PROJECT_STATE.md` only if the 300-output freeze genuinely succeeds.

This is the only authorized task for the current cycle.

---

# 18:02 HOURLY HEARTBEAT — 2026-09-24 +08:00

No meaningful new Codex completion has appeared since the 17:58 T072-AZ dispatch: main still ends at research-lead commit `cd05142020ea72d70ec1340a10eb30ba42603924`, `CODEX_TO_CHATGPT.md` still ends at the T072-AY re-entry diagnostic, and PR #219 has no Codex follow-up after the research-lead portability authorization.

Research-lead decision: **continue T072-AZ unchanged; do not create a new scientific direction or return to old-host polling.** The paid RTX 4090 path is already securely available and the only accepted activity is the minimal host-portability re-seal followed by the two frozen native-4K baseline runs.

Acceptance remains exactly: preserve source/checkpoint/config/cohort identities; change only GPU-name and absolute host paths; run one Retinexformer smoke and one SNR-Aware smoke; if both pass, immediately finish 150+150 outputs; freeze/hash all 300 outputs with native geometry/finiteness/runtime/peak-VRAM verification; keep `reference_reads=0` and `metrics=0`. Stop at the first genuine fail-closed blocker and do not improvise a workaround.

---

# MAIN-TABLE EXECUTION PROGRAM — locked 2026-09-24

The user has now explicitly authorized completing the full experiment program, with **comparative main-table completion as the first priority**. From this point until the comparative main table is complete, do not open side branches for in-domain rescue, extra method development, or nonessential analysis. The only scientific priority is the cross-domain / unseen-degradation main comparison.

## Paper question

Evaluate deployment robustness under source→target degradation shift. Source-domain performance is a reference/sanity check only. The headline question is whether target-free test-time adaptation improves robustness on unseen target degradation without target clean/reference supervision.

## Main-table target protocol

Use explicit source→target reporting rather than generic cross-dataset language.

Primary source-frozen target sequence:
- Source: accepted source training/checkpoint domain for each method, recorded explicitly in the table.
- Target-1: UHD-LL complete canonical held-out cohort — active now.
- Target-2: LSRW complete canonical paired test split once its authorized archive/protocol is available.
- Target-3: one additional paired real low-light target only after its canonical cohort/protocol is preregistered before result inspection. Do not improvise Target-3 from convenience.

For every target, target clean/reference is metrics-only and must remain sealed until **all methods included in that target's main table have their outputs frozen and hashed**.

## Main-table method set

### Fixed/source-frozen anchors
- Retinexformer — frozen accepted source checkpoint/config.
- SNR-Aware — frozen accepted source checkpoint/config.
- PromptIR — official frozen source model paired with the DCTTA comparison.

### TTA / cross-domain / zero-shot competitors
- PromptIR + DCTTA — principal direct TTA competitor; report the exact official source setting and adaptation granularity.
- MR. Illuminate — zero-shot/generalization competitor, if official code/checkpoint can be reproduced under the target protocol without target-reference leakage.
- QuadPrior — zero-reference competitor.
- ZERO-IG — per-image zero-shot competitor, if reproducible under the same target protocol.
- GM-MoE may be added as a generalization-oriented secondary competitor after the Tier-1 rows above are complete; it must not delay the Tier-1 table.

### Ours
- Ours-Step0 / Ours w/o TTT — same frozen method state before per-image adaptation.
- Ours-TTT — frozen T070-A per-image episodic adaptation.

Tier-1 rows that must be completed before the main table is considered minimally complete: Retinexformer, SNR-Aware, PromptIR, PromptIR+DCTTA, MR. Illuminate, QuadPrior, Ours-Step0, Ours-TTT. ZERO-IG is strongly preferred; GM-MoE is secondary.

## Main-table columns
At minimum record for every row:
- Method
- Source training / official source checkpoint setting
- Paradigm: fixed / domain-level TTA / zero-shot / per-image episodic TTT
- Target low images used for adaptation
- Adaptation granularity
- Target GT used during adaptation? — must be No for all valid rows
- Mean PSNR
- Median PSNR
- Mean RGB-SSIM
- paired delta versus the appropriate static/base counterpart when defined
- win fraction where defined
- adaptation time
- inference time
- peak VRAM
- test-time updated parameter/state size.

## Fairness / information boundary

- Never use target GT, clean/reference, PSNR/SSIM, baseline outcomes, or any reference-derived signal to choose checkpoints, hyperparameters, stopping, retry, repair, sample exclusion, or target-time state.
- Every method must use its source-frozen/officially declared source model; no target-domain supervised checkpoint substitution.
- All outputs for all main-table methods on a target must be frozen and hashed before target references are opened.
- If a method's official protocol is domain-level (e.g. DCTTA), clearly report target-low access and adaptation granularity; do not artificially force it into our per-image protocol.
- If official source-training settings differ across methods, disclose them explicitly. Do not imply matched source training where none exists.
- Ours-Step0 and Ours-TTT must share all scientific settings except the presence/absence of the frozen per-image TTT trajectory.

## Extended preregistration requirement

T072-L remains historical evidence for the original three-method UHD-LL analysis plan. Because the main-table method set is now expanded **before any UHD-LL reference outcome has been opened**, create a new task-owned extended analysis specification before any reference access. It must retain the existing UHD-LL metric implementation, complete 150-sample policy, bootstrap seed `20260922`, and fail-closed rules, while enumerating the expanded main-table rows and their immutable provenance. Do not overwrite T072-L; supersede it prospectively with an extended spec.

## Main-table execution order

1. **Finish the current T072-AZ task first**: paid-RTX4090 portability re-seal + Retinexformer/SNR-Aware native-4K smoke + 150×2 low-only output freeze. This remains the sole current task until completion.
2. Next hourly cycle: seal the expanded UHD-LL main-table analysis/provenance spec and prepare/run PromptIR + PromptIR+DCTTA output freeze on UHD-LL without references.
3. Following cycles: complete Ours-Step0, MR. Illuminate, QuadPrior, then ZERO-IG/Tier-2 rows, one scoped work package per hourly review.
4. Only after all declared UHD-LL main-table outputs are frozen and independently verified may the UHD-LL references be opened and the preregistered metrics/statistics computed.
5. Then repeat the same source→target protocol on LSRW, then the preregistered Target-3.
6. Only after the cross-domain comparative main table is complete move to ablations and analytical experiments.

## Current authorization

**Do not replace the current T072-AZ task. Continue T072-AZ exactly as already authorized.** The program above is the locked roadmap for subsequent hourly tasks, not permission to launch multiple new baselines concurrently in this cycle.

This roadmap exists to finish the complete experiment program while preserving one auditable, well-scoped task per research-lead cycle.

---

# 19:01 HOURLY HEARTBEAT — 2026-09-24 +08:00

No meaningful new Codex work has appeared since the main-table program was locked at `5ae4024224b9e3b355c00952a178944a21a902f9`: main contains no newer Codex commit, `coordination/CODEX_TO_CHATGPT.md` still ends at the T072-AY re-entry diagnostic, and PR #219 still has no Codex follow-up after the portability authorization.

Research-lead decision: **keep T072-AZ active unchanged. Old-host A6000 polling remains suspended.** The secure paid RTX 4090 path was already established; there is no justification for opening another scientific task until the portability re-seal and the two source-frozen UHD-LL baseline output sets are completed or a genuine new blocker is reported.

Current acceptance criteria remain: change only the GPU-name and host-specific absolute-path bindings; independently verify the frozen Retinexformer/SNR-Aware source-checkpoint-config identities and T072-I 150-low cohort; run one native-4K float32 smoke per baseline; if both pass, immediately complete all 150+150 outputs; freeze/hash exactly 300 outputs with geometry/finiteness/runtime/peak-VRAM evidence; keep `reference_reads=0` and `metrics=0`. No clean/reference access, no PSNR/SSIM, no target tuning, no new baseline, and no old-host polling in this cycle.

Single current task: **T072-AZ — paid-RTX4090 portability re-seal and RetinexFormer/SNR-Aware UHD-LL 150×2 output freeze.**

---

# 20:02 HOURLY HEARTBEAT — 2026-09-24 +08:00

No meaningful new Codex completion has appeared since the 19:01 checkpoint: main still has no Codex-authored commit newer than the prior T072-AY re-entry evidence, `coordination/CODEX_TO_CHATGPT.md` still ends at the same T072-AY RTX4090 gate diagnostic, and PR #219 still has no Codex follow-up after the explicit portability authorization.

Research-lead decision: **continue T072-AZ unchanged and keep the main-table program locked.** Do not reopen old-host A6000 polling, do not start DCTTA or any other baseline in parallel, and do not touch UHD-LL references. The current blocker is execution progress on the already-authorized RTX4090 portability re-seal, not a scientific-design blocker.

Acceptance remains unchanged: only GPU-name and host-path bindings may change; frozen RetinexFormer/SNR-Aware source-checkpoint-config identities and the exact T072-I 150-low cohort must be independently verified; run one native-4K float32 smoke per baseline; if both pass, immediately complete the 150+150 low-only outputs; freeze/hash exactly 300 outputs with geometry/finiteness/runtime/peak-VRAM evidence; maintain `reference_reads=0` and `metrics=0`.

Single current task: **T072-AZ — paid-RTX4090 portability re-seal and RetinexFormer/SNR-Aware UHD-LL 150×2 output freeze.**
---

# EXECUTE-NOW DIRECTIVE — 2026-09-24 20:56 +08:00

Codex: **start T072-AZ immediately. Do not wait for another hourly heartbeat.**

Use the already-established private key-based access to the paid RTX 4090 host. The sole objective remains: minimal host-portability re-seal → one RetinexFormer native-4K float32 smoke → one SNR-Aware native-4K float32 smoke → if both pass, immediately complete the full 150+150 low-only outputs and freeze/hash all 300 outputs.

Only GPU-name binding and host-specific absolute paths may change. Independently verify that all scientific source/checkpoint/config/T072-I cohort identities are unchanged before inference. Stop at the first genuine runtime/OOM/binding/geometry failure; no workaround, resize/tiling, precision rescue, checkpoint/config change, target tuning, reference access, or metric computation.

Completion evidence must include portability manifest/verifier, fresh GPU snapshot, both smoke receipts, full-run receipts if reached, exact 150/150 coverage per baseline, geometry/finiteness/runtime/peak-VRAM, per-output SHA256 manifest, `reference_reads=0`, and `metrics=0`.

This remains the only authorized task.
