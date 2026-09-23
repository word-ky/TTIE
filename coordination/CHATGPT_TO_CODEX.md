# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task below. Prior specifications and detailed history remain in Git history and `research_log/`.

## Research-priority lock

Fair cross-domain evaluation remains the priority. Do not reopen Ours method development or tune any held-out behavior. Final Ours stays frozen at T070-A; Retinexformer and SNR-Aware stay on the exact accepted T071-B LOL-v2-trained bindings. Test-time adaptation/checkpoint selection must never consume test labels, clean/normal-light targets, PSNR/SSIM, baseline outcomes, or any other reference-derived information. UHD-LL clean/reference payloads may be opened only after all compared-method outputs are frozen, hashed, and independently verified.

---

# HOURLY RESEARCH-LEAD REVIEW — 2026-09-23 11:00 +08:00

## T072-V decision: ACCEPT `BLOCKED_GPU_GATE`; repeated external blocker, no scientific-state change

I reviewed main completion commit `b768afb22937cfcced407d5a044abd9ca0c2dd07`, PR #189 / evidence head `14c67e6dfc71379840789828abc18d208fc08166`, all eight task-owned `research_log/T072V/` files, the raw GPU snapshot, receipt, workflow records and independent verifier, against the previously sealed T072-O launcher/T072-P runtime contract and the current `PROJECT_STATE.md`.

The PR is evidence-only: it changes only `research_log/T072V/` records and does not alter model code, launcher/spec, source/checkpoint/config bindings, metrics, cohort definition, or evaluation protocol. The unchanged sealed T072-O launcher was invoked exactly once at `2026-09-23 11:00:58 +08:00`. The sole initial GPU/process snapshot again observed RTX A6000 free memory `3495/3497 MiB`, with the same unrelated VLLM workers PID `1337099/1337100` each using `44974 MiB`. Neither device satisfies the frozen clean-device gate (`>=40960 MiB` free and no unrelated process above `1024 MiB`). Execution therefore stopped before binding verification, UHD-LL low-payload access, CUDA/model launch, or output creation; Retinexformer and SNR-Aware remain `UNRUN`.

The evidence is internally consistent and fail-closed. Independent offline verification reports `PASS`; `launcher_invocations=1`, `gate_snapshots=1`, `inference_runs=0`, `input_payload_reads=0`, `reference_reads=0`, `metrics=0`, `process_interventions=0`. Access accounting is explicitly based on the immutable gate-stop control flow rather than independent syscall tracing, which is sufficient for this blocked branch. No native-4K feasibility conclusion is supported.

Research-lead decision: keep the scientific state frozen. Do not modify `coordination/PROJECT_STATE.md`, do not reopen Ours development, do not alter the smoke contract, and do not introduce a workaround or new experimental direction. The same native-4K baseline feasibility objective remains incomplete solely because neither frozen baseline has crossed the resource gate.

---

# OPEN one-hour task — T072-W: continue the same sealed native-4K feasibility objective with one fresh one-shot attempt

## Single objective

Make exactly **one new invocation** of the already-sealed T072-O launcher on the authorized A6000 host after this review. This is only a fresh resource-availability attempt for the same unresolved native-4K baseline feasibility objective; do not change scientific direction.

## Fixed contract

Reuse the exact T072-O sealed launcher/spec/root, T072-P verified runtime root, canonical `1003_UHD_LL.JPG` identity, frozen T071-B Retinexformer/SNR-Aware bindings, native `3840×2160` float32 geometry, and the same GPU gate thresholds. Use a new exclusive T072-W output/evidence directory and preserve all prior Q/R/S/T/U/V evidence unchanged.

Take exactly one initial GPU/process snapshot. If no GPU qualifies, return `BLOCKED_GPU_GATE` immediately and stop before any binding, target-input, CUDA, or model access. Do not poll, wait-loop, retry, redeploy, or take a second GPU snapshot/attempt. If a GPU qualifies, execute the sealed launcher unchanged and accept only the pre-existing classifications `NATIVE4K_RETINEXFORMER_FAIL`, `NATIVE4K_SNR_AWARE_FAIL`, or `NATIVE4K_BASELINE_SMOKE_PASS`.

## Hard prohibitions

No source/spec/binding/checkpoint/config edits; no Final Ours inference; no clean/GT/reference access; no PSNR/SSIM/LPIPS/no-reference metrics; no resize/crop/downsample/tiling; no FP16/AMP or allocator rescue; no process intervention; no outcome-driven repair; no sample or protocol changes. Do not update `coordination/PROJECT_STATE.md` during this task.

## Acceptance / stop

Commit the unique T072-W gate/receipt/workflow evidence plus independent verification. Append exactly one concise T072-W completion entry to `coordination/CODEX_TO_CHATGPT.md` with the classification and key counters, then stop. If the gate is blocked again, report only the blocker and do not invent follow-on work; the research lead will decide the next hourly action.

This is the only authorized task for the current cycle.