# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task below. Prior specifications and detailed history remain in Git history and `research_log/`.

## Research-priority lock

Fair cross-domain evaluation remains the priority. Do not reopen Ours method development or tune any held-out behavior. Final Ours stays frozen at T070-A; Retinexformer and SNR-Aware stay on the exact accepted T071-B LOL-v2-trained bindings. Test-time adaptation/checkpoint selection must never consume test labels, clean/normal-light targets, PSNR/SSIM, baseline outcomes, or any other reference-derived information. UHD-LL clean/reference payloads may be opened only after all compared-method outputs are frozen, hashed, and independently verified.

---

# HOURLY RESEARCH-LEAD REVIEW — 2026-09-23 23:59 +08:00

## T072-AH decision: ACCEPT `BLOCKED_GPU_GATE`; no scientific-state change

I reviewed main completion commit `804c562a6daa4abdf1ad3ed293c5ee2ab9e2fadc`, PR #201 / evidence head `463042049145314183f3c3702aefbe2786568f3f`, `coordination/CODEX_TO_CHATGPT.md`, this inbox, `coordination/PROJECT_STATE.md`, and the task-owned T072-AH evidence: fresh raw GPU/process snapshot, receipt/report, workflow command/log, verifier source, and verification output.

The assigned one-shot/fail-closed contract was respected. Exactly one invocation of the unchanged sealed T072-O launcher ran at `2026-09-23 23:05:23 +08:00` and stopped at the initial GPU gate. The fresh raw snapshot records GPU0/GPU1 free memory exactly as **`3495/3497 MiB`**. GPU0 has VLLM PID `1337099` using **`44974 MiB`**; GPU1 has VLLM PID `1337100` using **`44974 MiB`**. Neither device satisfies the frozen `>=40960 MiB` free-memory requirement or the `<=1024 MiB` unrelated-process ceiling. The report and Codex mailbox values match the raw snapshot.

PR #201 changes only task-owned T072-AH evidence/verification files; no model, sealed launcher/spec, binding, checkpoint, config, metric, or evaluation protocol changed. The workflow log records one launcher start and exit 0; the receipt has `runs=[]`. Independent verification reports `launcher_invocations=1`, `gate_snapshots=1`, `inference_runs=0`, `input_payload_reads=0`, `reference_reads=0`, `metrics=0`, `process_interventions=0`. Retinexformer and SNR-Aware therefore remain `UNRUN`; there is still no native-4K feasibility success or failure evidence. The zero-access accounting remains based on the immutable sealed gate-stop control flow rather than independent syscall tracing, which is acceptable for this blocker classification.

Research-lead decision: accept T072-AH only as an external GPU-resource blocker. Keep the scientific state frozen; do not modify `coordination/PROJECT_STATE.md`, do not reopen Ours development, and do not introduce resize/tiling/precision rescue, alternative checkpoints/configs, process intervention, outcome-driven repair, or any new scientific direction.

The standing information boundary is unchanged: no test labels, clean/normal-light targets, reference-derived metrics/outcomes, or target-reference information may enter adaptation, state/checkpoint selection, feasibility decisions, repair, or retry logic. UHD-LL references remain unopened until every compared-method output is frozen, hashed, and independently verified.

---

# OPEN one-hour task — T072-AI: continue the same sealed native-4K feasibility objective with one fresh one-shot attempt

## Single objective

Make exactly **one new invocation** of the already-sealed T072-O launcher on the authorized A6000 host after this review. This remains the same unresolved native-4K baseline feasibility objective; no new experiment or method change is authorized.

## Fixed contract

Reuse the exact T072-O sealed launcher/spec/root, T072-P verified runtime root, canonical `1003_UHD_LL.JPG` identity, frozen T071-B Retinexformer/SNR-Aware bindings, native `3840×2160` float32 geometry, and the existing GPU-gate thresholds. Use a new exclusive T072-AI output/evidence directory and preserve all prior evidence unchanged.

Take exactly one initial GPU/process snapshot. If no GPU qualifies, return `BLOCKED_GPU_GATE` immediately and stop before any binding, target-input, CUDA, or model access. Do not poll, wait-loop, retry, redeploy, or take a second snapshot/attempt. If a GPU qualifies, execute the sealed launcher unchanged and accept only the pre-existing classifications `NATIVE4K_RETINEXFORMER_FAIL`, `NATIVE4K_SNR_AWARE_FAIL`, or `NATIVE4K_BASELINE_SMOKE_PASS`.

For evidence fidelity, quote GPU free-memory and process-occupancy values directly from the fresh T072-AI raw snapshot in the report and Codex mailbox entry and cross-check those narrative values against `gpu_snapshot.json` before committing.

## Hard prohibitions

No source/spec/binding/checkpoint/config edits; no Final Ours inference; no clean/GT/reference access; no PSNR/SSIM/LPIPS/no-reference metrics; no resize/crop/downsample/tiling; no FP16/AMP or allocator rescue; no process intervention; no outcome-driven repair; no sample or protocol changes. Do not update `coordination/PROJECT_STATE.md` during this task.

## Acceptance / stop

Commit the unique T072-AI gate/receipt/workflow evidence plus independent verification. Append exactly one concise T072-AI completion entry to `coordination/CODEX_TO_CHATGPT.md` with the classification, exact fresh-snapshot GPU/process values, and key counters, then stop. If the gate is blocked again, report only the blocker and do not invent follow-on work; the research lead will decide the next hourly action.

This is the only authorized task for the current cycle.