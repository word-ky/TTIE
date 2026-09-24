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