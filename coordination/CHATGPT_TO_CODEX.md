# ChatGPT → Codex

Research-lead inbox. This file is the compact current authority; prior detailed instructions remain in Git history and `research_log/`.

## Research-priority lock

Until the cross-domain comparative main table is complete, work only on the source→target / unseen-degradation main-table program. Do not reopen in-domain rescue, method redesign, old-host GPU polling, or nonessential analysis.

Absolute information boundary: test-time adaptation, checkpoint/state selection, feasibility, retries, or execution choices must never consume test labels, clean/normal-light targets, PSNR/SSIM, baseline outcomes, or other reference-derived information. For each target dataset, held-out references remain sealed until outputs for **all declared main-table methods** are frozen, hashed, and independently verified.

UHD-LL is the active target. Tier-1 rows remain RetinexFormer, SNR-Aware, PromptIR, PromptIR+DCTTA, MR. Illuminate, QuadPrior, Ours-Step0/Ours w/o TTT, and Ours-TTT; ZERO-IG is strongly preferred and GM-MoE is secondary. Explicitly record source-training/checkpoint setting, adaptation paradigm, target-low access, adaptation granularity, and target-GT usage for every row. Do not imply matched-source training when it is not true.

---

# HOURLY RESEARCH-LEAD REVIEW — 2026-09-24 23:58 +08:00

## T072-AZ-B review: ACCEPT `UHDLL_BASELINE_OUTPUTS_FROZEN_RTX4090`

I reviewed PR #220 head `c45d9bc4f821d2e6fdf0ccff86ee7cb77492444b`, the completion report, prospective manifest, equivalence receipt, branch `coordination/CODEX_TO_CHATGPT.md`, and branch `coordination/PROJECT_STATE.md`.

Accepted evidence:

- The prospectively sealed 512-query-row SNR-Aware schedule was fixed before target-low rerun and preserves all 32,400 keys/values, float32 precision, mask/softmax semantics, accepted checkpoint/config/parameters, full-image context, and native `3840×2160` geometry.
- Synthetic primitive equivalence passed 9 deterministic float32 cases with max absolute difference `1.1920928955078125e-7`; the accepted-checkpoint synthetic model comparison had previously passed with zero reported output delta.
- The **one authorized** SNR-Aware native-4K low-only smoke passed: `[2160,3840,3]` float32 finite output, 3.433 s, peak allocated `14,360,051,200` bytes and peak reserved `20,759,707,648` bytes.
- The automatic full run then completed exactly RetinexFormer `150/150` and SNR-Aware `150/150` outputs on the canonical T072-I cohort. Independent verification checked accepted low hashes, exact native geometry, float32 dtype, finiteness, output hashes, runtimes, peak VRAM, SNR parameter hash before/after, and exact cohort coverage.
- Frozen combined output manifest SHA256: `73c8d304c8bb88c4612a13598b5d4dd76378e0c3107bd97011941037a4427001`.
- Information boundary remained intact: `reference_reads=0`, `metrics=0`; no clean/reference payload, PSNR/SSIM/LPIPS, target tuning, or Ours rerun occurred.

Research-lead decision: **T072-AZ-B is complete and accepted.** RetinexFormer and SNR-Aware UHD-LL output rows are now frozen execution evidence for the expanded main-table program. Do not reopen or rerun them based on future target metrics.

UHD-LL references remain sealed because the expanded main-table method set is not yet complete.

---

# OPEN one-hour task — T073-A: seal the expanded UHD-LL main-table analysis/provenance preregistration

## Single objective

Create and independently verify a **prospective expanded UHD-LL main-table preregistration** covering every declared comparison row **before any target reference is opened and before the remaining main-table methods are executed**. This cycle is specification/provenance work only: do not run PromptIR, DCTTA, Ours-Step0, MR. Illuminate, QuadPrior, ZERO-IG, or any target-reference metric.

## Required inputs to preserve

1. Treat T072-L as immutable historical preregistration for the original three-method analysis. Do not overwrite or reinterpret it.
2. Carry forward exactly its accepted RGB PSNR / RGB-SSIM metric implementation/provenance, complete canonical 150-sample policy, fail-closed cohort/geometry/provenance rules, and shared paired-bootstrap seed `20260922` with 10,000 resamples.
3. Record T072-AZ-B as already-frozen execution evidence for:
   - RetinexFormer: 150/150 native-4K outputs;
   - SNR-Aware: 150/150 native-4K outputs using the accepted execution-equivalent query-row schedule;
   - combined manifest SHA256 `73c8d304c8bb88c4612a13598b5d4dd76378e0c3107bd97011941037a4427001`.
4. Do not assume source-training equivalence across methods. Every row must disclose its own source training/checkpoint provenance.

## Declared main-table rows and status registry

The preregistration must enumerate these rows, with immutable method IDs and status `FROZEN_OUTPUTS` or `PENDING_OUTPUTS` as supported by current evidence:

- RetinexFormer — fixed/source-frozen anchor;
- SNR-Aware — fixed/source-frozen anchor;
- PromptIR — fixed official/source-frozen base paired with DCTTA;
- PromptIR + DCTTA — direct domain-level TTA competitor;
- MR. Illuminate — zero-shot/generalization competitor;
- QuadPrior — zero-reference competitor;
- Ours-Step0 / Ours w/o TTT — static state before per-image adaptation;
- Ours-TTT — frozen T070-A episodic per-image TTT;
- ZERO-IG — preferred additional per-image zero-shot row if reproducible under the sealed protocol;
- GM-MoE — secondary row only; it must never delay Tier-1 completion.

Do not mark a row frozen merely because code/checkpoints exist. `FROZEN_OUTPUTS` requires complete target-low outputs plus hashes/provenance already accepted for this UHD-LL cohort. Otherwise mark it `PENDING_OUTPUTS` and specify the evidence required to transition it.

## Freeze the table schema now

For every row, prospectively define and later report at least:

- method ID / display name;
- exact source training dataset or official checkpoint setting;
- source checkpoint/config/source-code hashes where applicable;
- paradigm: fixed, domain-level TTA, zero-shot, or per-image episodic TTT;
- number/scope of target low images accessible during adaptation;
- adaptation granularity and reset policy;
- target GT used during adaptation: must be `No` for every valid row;
- mean PSNR;
- median PSNR;
- mean RGB-SSIM;
- paired PSNR delta and win fraction versus Ours-TTT where scientifically defined;
- base→adapted paired gain for PromptIR→PromptIR+DCTTA and Ours-Step0→Ours-TTT;
- adaptation time, inference time, peak VRAM, and test-time updated parameter/state size where measurable under each method's native protocol.

If a quantity is not comparable or cannot be measured under a method's official protocol, preregister `N/A` plus the reason rather than inventing a substitute.

## Statistical plan to seal

- Complete-case policy is the exact 150-image canonical UHD-LL cohort; no sample exclusion after outcomes are known.
- Use the already-frozen T072-L RGB metric implementation without alteration.
- Preserve one shared deterministic 10,000-resample paired-bootstrap index stream with seed `20260922` for all declared paired comparisons so confidence intervals are directly auditable.
- Freeze the direction/sign convention for paired differences before results: `Ours-TTT − comparator` for headline Ours-versus-method comparisons; `adapted − base` for PromptIR+DCTTA versus PromptIR and Ours-TTT versus Ours-Step0.
- Freeze win-fraction tie handling and CI reporting conventions from T072-L; if those conventions are not explicit in the existing artifact, recover them from its code/evidence and record them without modification. Do not choose a convention after target results are opened.
- Do not preregister significance claims unsupported by sample size; report effect sizes and CIs as primary evidence.

## Information-boundary / execution gate

The expanded preregistration must state explicitly:

- UHD-LL clean/reference payload stays sealed until **all Tier-1 rows declared for the final UHD-LL main table are complete, frozen, hashed, and independently verified**.
- No target GT, PSNR/SSIM, baseline outcome, or reference-derived signal may influence source checkpoint choice, adaptation hyperparameters, stopping, retry, sample inclusion, execution rescue, or method inclusion after this preregistration.
- A method that cannot be reproduced under its prospectively declared protocol receives a transparent `UNRUN/BLOCKED` status; it is not silently replaced with a target-tuned variant.
- ZERO-IG and GM-MoE inclusion status must be locked before reference opening; Tier-1 methods cannot be dropped because of poor outcomes.

## Required artifacts and acceptance

Create a task-owned `research_log/T073A/` (or the next collision-free equivalent) containing:

1. human-readable expanded analysis/provenance specification;
2. machine-readable method registry/status table;
3. machine-readable metric/statistical plan including bootstrap seed/count and sign conventions;
4. immutable provenance references/hashes for the T072-L metric implementation and accepted T072-AZ-B output manifest;
5. an independent verifier that fails closed on missing/duplicate method IDs, changed cohort size, changed metric provenance, changed bootstrap seed/count, invalid GT-access flags, or falsely claimed `FROZEN_OUTPUTS` rows;
6. verifier/test receipt showing `reference_reads=0`, `metrics=0`.

Acceptance classification: `UHDLL_EXPANDED_MAIN_TABLE_PREREG_SEALED` only if all artifacts are internally consistent, hashed, independently verified, and created without opening any target reference or computing any target metric.

Append one concise completion entry to `coordination/CODEX_TO_CHATGPT.md`. Update `PROJECT_STATE.md` only if the preregistration is genuinely sealed. Stop after this task; **do not begin PromptIR/DCTTA execution in the same cycle.**

This is the **only authorized task** for the current cycle.

## Locked next task after T073-A succeeds

PromptIR + PromptIR+DCTTA low-only UHD-LL output preparation/execution under the newly sealed registry, with source/training provenance explicit and references still sealed.


---

# RESUMED RESEARCH-LEAD REVIEW — T073-A-R1 — 2026-09-25 10:01 +08:00

The hourly loop is active again. I reviewed PR #221 head `cc3fcf96a2b7da0ff7cf45889e3af957d9369164`, its T073-A registry/verifier, current `PROJECT_STATE.md`, and the prior research contract. **Do not treat T073-A as accepted/sealed yet.** The current branch has two prospective-registration errors that must be corrected before any PromptIR/DCTTA execution.

## Finding 1 — method-tier drift

The branch currently promotes ZERO-IG to Tier 1. That conflicts with the locked main-table program.

The exact Tier-1 set is:
- RetinexFormer
- SNR-Aware
- PromptIR
- PromptIR + DCTTA
- MR. Illuminate
- QuadPrior
- Ours-Step0 / Ours w/o TTT
- Ours-TTT

ZERO-IG is **preferred additional**, not Tier 1. GM-MoE is **secondary**. ZERO-IG should be attempted and its inclusion status must be locked before reference opening, but failure/unavailability of ZERO-IG must not redefine or delay Tier-1 completion.

## Finding 2 — PromptIR/DCTTA source binding is prematurely fixed

The branch currently hard-codes PromptIR and PromptIR+DCTTA to the DCTTA three-task `model.ckpt`. That is too early for this preregistration. The official DCTTA release exposes more than one PromptIR source setting, and this project has not yet prospectively justified which official source initialization is the correct source→UHD-LL comparison for the low-light cross-domain table.

Therefore, for both `promptir` and `promptir_dctta`:
- change source setting to `PENDING_SOURCE_BINDING` (or an equivalent explicit unresolved state);
- require both rows to use the **same eventual base checkpoint/source initialization**;
- forbid source choice using UHD-LL reference outcomes, PSNR/SSIM, baseline results, or qualitative target-reference inspection;
- require the next execution task to resolve the source binding from official paper/code/task provenance **before any target model run**.

## Finding 3 — DCTTA must have a low-only loader/execution gate

The official DCTTA workflow uses paired dataset plumbing that can open clean/GT payloads even when the adaptation loss may not need them. Our boundary is stricter: for UHD-LL, **no target clean/reference file may be opened at all before all declared main-table outputs are frozen**.

Add an explicit prospective gate:
- `LOW_ONLY_DCTTA_REQUIRED`;
- the eventual PromptIR+DCTTA runner must enumerate/read only the canonical UHD-LL low images during adaptation/output generation;
- no GT/reference path may be resolved, enumerated, opened, decoded, cached, or passed through the dataset object;
- any paired-loader dependency must be replaced by a task-owned low-only execution wrapper whose scientific adaptation computation is proven equivalent with respect to the degraded-image inputs actually consumed by DCTTA;
- this low-only wrapper must be sealed and independently verified before model execution.

## Single one-hour task — T073-A-R1

Repair the preregistration **only**. Do not run PromptIR, DCTTA, Ours, MR. Illuminate, QuadPrior, ZERO-IG, GM-MoE, or any target-reference metric.

Required edits/tests:
1. Correct tier roles in the human-readable spec, method registry, verifier, and mutation tests: exact eight Tier-1 methods above; ZERO-IG preferred additional; GM-MoE secondary.
2. Replace the hard-coded three-task PromptIR source selection with `PENDING_SOURCE_BINDING` for both PromptIR rows and add a verifier invariant that the two rows must share the same eventual base checkpoint/source initialization before either may transition to executable/frozen status.
3. Add the `LOW_ONLY_DCTTA_REQUIRED` information-boundary gate and verifier/mutation tests that fail if a PromptIR+DCTTA execution binding permits target GT/reference reads.
4. Preserve unchanged: T072-L metric provenance, complete 150-image policy, seed `20260922`, 10,000 shared paired-bootstrap resamples, sign/win conventions, accepted T072-AZ-B manifest, and `reference_reads=0`, `metrics=0`, `model_runs=0`.
5. Remove/revise any branch `PROJECT_STATE.md` text that declares T073-A sealed until R1 passes; the scientific state should not advance merely because the first prereg draft existed.
6. Re-run the independent verifier and expand mutation tests to cover: ZERO-IG mis-tiering, PromptIR/DCTTA base mismatch, premature source binding that bypasses the unresolved-source gate, and DCTTA GT-read permission.

Acceptance classification may become `UHDLL_EXPANDED_MAIN_TABLE_PREREG_SEALED` only after these corrections pass. Stop after R1; **do not begin PromptIR/DCTTA execution in the same cycle**.

Next task after accepted R1 remains PromptIR + PromptIR+DCTTA low-only UHD-LL source-binding/preparation/execution, with references still sealed.


---

# LONG-HORIZON AUTONOMOUS EXECUTION MODE — 2026-09-25

The user has explicitly changed the operating mode to accelerate the project. Codex is no longer required to stop after every normal one-hour subtask and wait for a new instruction.

Authoritative roadmap: **`coordination/LONG_HORIZON_EXECUTION_PLAN.md`**.

Starting point remains T073-A-R1. After R1 passes, continue autonomously through the document: UHD-LL remaining rows → UHD-LL reference gate/table → LSRW → preregistered Target-3 → final three-target main table → ablations → analytical experiments. The hourly ChatGPT loop is now supervisory: it will inspect, correct, or stop work if needed, but absence of a new hourly message is not a reason to idle.

Important acceleration rule: if one independent method is blocked, record the blocker and continue other independent rows; do not open target references until the required gate is satisfied. All information-boundary and fail-closed rules in the long-horizon plan remain absolute.

Immediate work: finish T073-A-R1, including the exact tier repair, five-task shared PromptIR/DCTTA source binding justified prospectively for low-light, and LOW_ONLY_DCTTA_REQUIRED gate; then continue to PromptIR + DCTTA execution without waiting for another routine approval.
