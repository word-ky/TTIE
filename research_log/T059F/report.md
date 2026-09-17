# T059-F — PARTIAL: mandatory invariance stop

Exact requested nonnegative scale makes three nonconstant banks constant. We stop without either scientific classification, as required by the invariance stop rule. The accepted T059-E negative remains unchanged. No rerun, repair, training, calibration or rollout.

Source: `62cd75f0` on `codex/T059F-positive-scale`; authorization `b546c1992ffe3ba18d785e64915eeea9e1282e4c`. Sole A6000-server CPU audit `20260918-030750-ttie-t059f-scale`, 2026-09-17 19:07:54–19:08:00 UTC, exit 0 (successful audit, scientific PARTIAL). Local tests 3 passed in 4.33s; server 3 passed in 1.49s. Separate verifier PASS, independently reproducing scalars, losses and the three invariance failures.

Only five SHA-bound E artifacts were opened: checkpoint (hash only), split metadata, accepted completion metadata, heldout statistics and heldout row tensors. All before/after hashes agree. Checkpoint SHA256 e15e91c4e9be401bcac6d3de039ec2d1ae7141fb6cec40c388d22549652d43d0. All 1529 rows / 80 banks match E inner-held indices; excluded outer supervision never opened. Source bindings unchanged. All requested counters zero.

Replay: relative Huber 0.22107574343681335; median Spearman 0.9356521739130435 and median regret 0 recomputed exactly, including every bank. Detail 0.868874192237854 / 0.4915534257888794 and legacy 0.9570105671882629 / 0.9057643413543701 are exact readbacks from hash-bound accepted E statistics. These directional aggregates were NOT independently regenerated: E row payload does not persist cosine vectors, and F forbids new forwards/gradient generation.

Provisional computed aggregate corrected Huber 0.21471332013607025, fixed threshold 0.07650849781930447, margin -0.13820482231676579. This number is retained as stopped-run evidence, not an accepted task classification. All 80 scales, per-bank losses and worst 10 are in result.json. Scale min/p10=0, median=0.8545731902122498, mean=0.7311551291495562, p90=1.2387234449386597, max=2.1135330200195312. Zero-scale banks=20 (17 singletons plus 3 nonconstant banks).

|Bank|dot(rp,rt)|dot(rp,rp)|Original Spearman|Original regret|After zero scale|
|---|---:|---:|---:|---:|---|
|230|-0.5237829685211182|0.023528557270765305|-0.2678260869565217|4.667080879211426|Spearman undefined, regret 0|
|280|-11.242471694946289|0.2774660587310791|-0.9381263616557736|6.364080429077148|Spearman undefined, regret 0|
|305|-9.574722290039062|0.10522497445344925|-0.391304347826087|6.284384727478027|Spearman undefined, regret 0|

Each bank has 24 rows and a unique state-0 anchor. Their exact negative dot products force a=0 under max(0,dot/den). This is a mathematical boundary of nonnegative versus strictly positive scaling, not a floating-point sign ambiguity. Stable average ties and earliest argmin remain the historical conventions. No epsilon floor, exclusion, sign flip or alternate rule was introduced. The task labels any invariance change an implementation error; we honor its stop, but evidence locates the cause in its prescribed zero-scale case rather than an arithmetic implementation mismatch. Research lead should adjudicate zero-scale semantics in a new task before further diagnosis.

Commands: `python -m pytest research_log/T059F/test_core.py -q -p no:cacheprovider --import-mode=importlib`; `python -m research_log.T059F.run --input <frozen E artifacts> --output <run>/artifacts/T059F`; `python research_log/T059F/verify.py <run>/artifacts/T059F`. CPU float32 dot, residuals and Huber preserve E conventions; no model inference occurs. One preliminary PowerShell read-only metadata command had a quoting error; corrected using local source before the sole audit, with no scientific access or rerun.
