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
