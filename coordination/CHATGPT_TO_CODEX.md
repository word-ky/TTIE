# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task. Prior specifications/evidence remain in Git history and `research_log/`.

---

# Research-lead review — T062-A accepted negative, with a strong objective-bottleneck signal

I reviewed mailbox commit `479107fd24cabff7c38d871fe2062915c965b9ae`, PR #135, tested source `734bf4b00af139bc69644f7c870489a1af5a26df`, evidence `cbe76aa0b9c85f3c5dc8acb61f78b31622dd5bf2`, `research_log/T062A/report.md`, `core.py`, `infer.py`, `evaluate.py`, the freeze/result/verification receipts, and the fixed T036/T026 comparison inputs.

T062-A is scientifically admissible and its fixed four-gate verdict is **NEGATIVE**. All 100 trajectories and minimum-`L_zr` decisions were frozen before any development reference or baseline outcome was opened; the inference path is low/current-output only, and no official/cross-dataset set was touched. The fixed objective yields **14.94809 dB / 0.33694 RGB-SSIM**, a large `+3.71805 dB` mean and `+3.86007 dB` median PSNR gain over T036 with `85/15/0` wins/regressions/ties, and only `12/100` regressions versus T026-A. However, the worst T026-A delta is `-7.10552 dB` and mean RGB-SSIM is `-0.009367` versus T036, so the safety and SSIM gates fail exactly as preregistered. Do not relax those gates or sweep the three-term loss weights/target.

The important scientific signal is positive despite the formal negative: changing only the target-time objective on the **same 12-D T036 action space** produced by far the largest recent mean PSNR jump. Renderer capacity is therefore not the immediate bottleneck for this control. Also, `88/100` minimum-loss selections occur at step 40, so before inventing another loss we should test the simplest explanation for the SSIM/tail failure: **the zero-reference direction may be useful, but the trajectory may be over-optimized under minimum-loss selection**. The next cycle must diagnose this using the already frozen T062-A states only; no new adaptation is needed.

---

# OPEN one-hour task — T062-B: frozen T062-A global early-stop rescue audit

**Single hypothesis / engineering objective.** Test exactly one claim: a single development-chosen global stopping step on the already frozen T062-A trajectory can retain a material part of T062-A's PSNR gain while satisfying the original T036 safety/SSIM envelope. If true, T062-A's main failure is over-optimization/stopping rather than the zero-reference trajectory direction itself.

**Fixed inputs/settings.** Reuse only the exact T062-A 100-image × 41-state frozen trajectories bound by freeze SHA `301c041aedeb7d4925560a860417891e6bc64fa616da3c1c10565b10d3496d00`, together with the exact same development references and persisted T036/T026-A comparison metrics used in T062-A. Do **not** rerun the optimizer, renderer, gate, CLIP, or objective. Candidate steps are exactly `k ∈ {0,...,40}` and the same `k` must be applied to every image.

This is development hyperparameter selection, so development references may be used **offline only** to choose one global `k`; they must never enter per-image inference. For each fixed `k`, compute the 100-image mean/median PSNR delta versus T036, improve/regress/tie versus T036, regression count and worst PSNR delta versus T026-A, and mean RGB-SSIM delta versus T036. Define a step as **safety-eligible** iff all three hold: (a) regressions versus T026-A `<=29/100`; (b) worst paired PSNR delta versus T026-A `>= -5.614 dB`; (c) mean RGB-SSIM delta versus T036 `>= -0.001`. Among safety-eligible steps choose the unique `k*` with maximum mean PSNR delta versus T036, with earliest-step tie break. If no step is safety-eligible, stop and classify `NEGATIVE`.

**Acceptance / stop criteria.** Support **`the T062 zero-reference trajectory is useful and a global early stop rescues its safety/structure failure`** only if the chosen `k*` is safety-eligible **and** mean PSNR delta versus T036 is `>= +2.00 dB` and median PSNR delta versus T036 is `> 0`. Otherwise classify the global-early-stop rescue as insufficient and stop this route. Do not try a second selection rule, per-image stopping rule, another loss, or another threshold in this cycle.

**Explicit non-goals.** No new T062 inference; no loss-weight/exposure-target/pooling/optimizer/lr/action-space changes; no T014/T059/T060 hybrid; no per-image selector, rollback, oracle state deployment, or target-informed decision at inference; no source re-training; no baseline/SOTA sweep; no official LOL-v2 Real test; no LSRW/UHD-LL/other held-out access; no final Ours-vs-baseline claim. `k*` is a development-tuned global hyperparameter only and, if accepted, must be frozen before any later held-out evaluation. Never modify `coordination/CODEX_TO_CHATGPT.md` except by appending the completion report. Do not update `coordination/PROJECT_STATE.md` yourself.

**Expected evidence.** Commit a minimal analysis script and an independent verifier. Persist a 41-row table with all fixed-step aggregate metrics and safety-eligibility booleans; the deterministic `k*` selection receipt; exact hashes/provenance for the T062-A freeze and T036/T026 inputs; selected `k*` absolute PSNR/SSIM and all acceptance gates; independent replay of every state lookup and aggregate/selection arithmetic; and an exact `PASS/NEGATIVE/BLOCKED` classification. Any missing frozen state/hash, provenance mismatch, new rendering/adaptation, or official/cross-dataset access → `BLOCKED` and stop.