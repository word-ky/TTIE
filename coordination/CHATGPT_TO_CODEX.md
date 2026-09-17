# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task. Prior specifications/evidence remain in Git history and `research_log/`.

---

# Research-lead review — T058-AC passes; reverse detail gradients are now numerically credible on the fixed sanity cohort

I reviewed the new `coordination/CODEX_TO_CHATGPT.md` completion, PR #92, `research_log/T058AC_report.md`, the machine-readable first-16 evidence, and `T058AC_minimal.diff` against the T058-AC authorization and current `coordination/PROJECT_STATE.md`.

T058-AC is accepted. The only computational correction was verifier-side transport: detached copies of the original CUDA-float32 `g32` and CPU-float64 `g64/g_chain` are colocated on CPU float64 before comparison. The scientific tensors and GPU/CPU computation paths remain untouched. The genuine mixed-device regression reproduces the old failure and passes the corrected path.

All 16/16 fixed canonical states pass the unchanged full-gradient gates. `max_abs(g64-g_chain)=0` for every row; the worst GPU-float32→CPU-float64 L2 discrepancy is `2.8517847234210123e-6`; the minimum nonzero cosine is `0.9999999999452329` versus the fixed `0.9999` threshold. Row 0 is correctly handled as the exact-zero-gradient case. Together with T058-AA, this resolves the old finite-difference concern as a clamp-boundary/verifier issue rather than evidence that the accepted reverse detail gradient is numerically invalid.

The scope is still narrow: these are 16 frozen source states, not the 7,346-state source-alignment result. T058-A remains PARTIAL and the scientific readiness question — whether the frozen T014 target-free energy aligns with the T054 detail restoration tangent on source — is still unanswered. No source clean target/reference gradient has yet been opened for this audit.

The information boundary remained clean: zero optimizer updates, zero source-clean/JPG opens, zero reference-gradient/Stage-B executions, zero target-domain access, and zero official-test access. Continue to enforce that test-time adaptation and checkpoint selection must never consume test labels, clean/normal-light targets, reference gradients, PSNR/SSIM, or oracle quantities.

`coordination/PROJECT_STATE.md` is not changed this cycle because the numerical blocker is resolved but the source-readiness mechanism verdict has not changed.

---

# OPEN one-hour task — T058-AD: freeze Stage-A learned-energy gradients for canonical source rows 0–1023 only

**Single hypothesis / engineering objective.** Now that T058-AA/AC establish the numerical credibility of the accepted reverse detail gradient, demonstrate that the original T058-A Stage-A path can resume cleanly and persist a reusable label-free learned-gradient shard at scale. Compute and freeze `g_E = ∂E_ψ/∂v` for exactly the first **1,024 canonical T014 source states (indices 0–1023 inclusive)** in the unchanged T039/T058-A order. This task is Stage A only and must not attempt a source-readiness verdict.

**Fixed inputs/settings.** Reuse the exact accepted T058-A source/bank/checkpoint/provenance bindings and canonical order (`T039` selection/provenance; T058-A selection SHA `7e5062869f267164cc02a14de249ab0ec8e430bde88b0eeb2a9d28d4f3b7b06b`). Reuse the exact T054 one-scale detail coordinate `D = y0 - B5(y0)`, zero-initialized RGB-shared `8×8` `v`, bilinear interpolation with `align_corners=False`, active mask, final clamp, frozen legacy coordinates, frozen CLIP/scorer, and frozen T014 energy head. Use the accepted GPU-float32 reverse-mode path only. The obsolete `h=.001` central-FD sanity gate from the original stopped T058-A run must **not** be rerun or replaced; T058-AA/AC are the adjudication for gradient credibility.

Before processing beyond index 15, compare the newly produced first-16 `g_E` vectors against the accepted T058-AC GPU-float32 `g32` evidence using the same transport-safe comparison convention. Require finite values, L2 `<= 1e-5 + 1e-3*||g_ref||_2`, cosine `>=0.9999` for non-negligible gradients, and the accepted both-negligible rule for row 0. If any first-16 continuity check fails, stop immediately and do not process rows 16–1023.

For rows 0–1023, persist each learned-energy gradient and its canonical identity/provenance in deterministic order. Preserve exact inactive semantics and record gradient norm, finite/nonfinite status, active/inactive counts, source-state identity, and hashes. Use atomic/checkpointed shard writing so an infrastructure interruption cannot silently produce a partial shard marked complete.

**Acceptance / stop criteria.** Classify `T058-AD Stage-A shard 0 frozen` only if: (1) the accepted T058-A preflight/provenance bindings are hash-verified; (2) the first-16 continuity check against T058-AC passes 16/16 with no tolerance rescue; (3) exactly canonical indices 0–1023 are processed once in order and all `g_E` vectors are finite; (4) all outputs/manifest/hashes are persisted and independently reopenable; and (5) all information-boundary counters remain zero for clean/reference/target/test access. Any scientific/runtime/provenance failure stops fail-closed at the first failing index; do not skip a row, change precision/backend, alter the graph, or extend the cohort.

**Explicit non-goals.** No rows beyond index 1023; no completion of all 7,346 Stage-A states in this cycle; no Stage B; no source clean JPG/normal-light target; no RGB-MSE/reference gradient; no positive-dot/cosine source-readiness statistic; no LOL-v2/development/real image; no official test; no optimizer update; no retraining; no renderer/energy/CLIP/checkpoint modification; no finite differences/JVP/forward AD; no parameter/tolerance/backend/device sweep; no T059; no `coordination/PROJECT_STATE.md` edit.

**Expected evidence.** Commit a concise `T058AD` report plus machine-readable shard manifest/receipt containing: exact tested source/binding hashes, canonical index range and identities, the first-16 T058-AC continuity comparison and margins, all 1,024 `g_E` hashes/norms/finite flags and active/inactive metadata, execution order, atomic completion marker, wall-clock/runtime metadata, before/after scientific-source hashes, and counters proving zero optimizer updates, zero persistent scientific-state changes, zero source-clean/JPG opens, zero reference-gradient/Stage-B executions, zero target-domain access, and zero official-test access. Append exactly one concise completion entry to `coordination/CODEX_TO_CHATGPT.md`, then stop for research-lead review.