# T059-F2 — DONE

`T059-E value failure is not explained by bankwise positive-scale miscalibration after offset removal`.

Under the authorized least-squares scale / continuous Huber-loss-limit diagnostic, Huber is **0.21471332013607025**, above the frozen **0.07650849781930447** ceiling (margin **-0.13820482231676579**). It equals the T059-F stopped arithmetic exactly; no epsilon or alternate scale rule was introduced. This diagnostic does not reverse T059-E's negative verdict or authorize rollout, and it is not a search over Huber-optimal scales.

Authorization abbeec6015d456235ae60d47571fe05eb6b61738; tested source 5ba2ec7fcff70531505f206a71a707eeecbb2fdb. Sole CPU audit 20260918-042133-ttie-t059f2-limit ran 2026-09-17 20:21:38–20:21:41 UTC, exit0. Server focused tests: 3 passed in 1.44s. Separate verifier PASS, independently recomputing scales, piecewise Huber, ranks/regrets and classification without importing the F2 implementation.

All 1529 rows / 80 banks are retained: 60 positive-scale banks; exactly boundary banks 230/280/305; 17 degenerate singleton banks (5,6,30,55,80,82,83,105,130,155,180,205,255,282,330,355,380). Positive branches have a=q; boundary and degenerate admissible_scale fields are null, never zero. Degenerate banks contribute their deterministic losses but provide no scale evidence.

For every a>0, sign(a*(p_i-p_j))=sign(p_i-p_j). Therefore all pairwise ordering, ties and earliest argmin stay unchanged; taking the loss limit as a approaches zero from above does not replace the ordering with the ordering of a constant vector. The implementation evaluates zero only as the continuous Huber loss limit, and obtains ranks/regrets from original predictions. Original per-bank ranks/regrets replay exactly, median Spearman 0.9356521739130435 and median regret 0. The negative rankings and original large regrets on the three boundary banks remain intact.

All 80 F numerators/denominators and per-bank corrected losses replay exactly. E relative Huber replays exactly at 0.22107574343681335. E directional aggregates are hash-bound recorded readbacks, not freshly regenerated gradients. The exact E checkpoint hash is e15e91c4e9be401bcac6d3de039ec2d1ae7141fb6cec40c388d22549652d43d0. E split, rows, statistics, checkpoint (hash only) and F result are the only scientific input files; their before/after hashes and six source bindings are retained in result.json. No C2 outer supervision was read. All training/optimizer/model-forward/new-image/new-feature/new-Jacobian/new-reference/outer/target/LOL-v2/official-test/inference-reference counters are zero.

All per-bank losses, partitions, raw ratios and worst10 are in result.json. Historical E/F code and evidence were not modified. No model/head execution, training, rerun, per-bank inference calibration or real-domain experiment occurred.

Operational failures: D disk ENOSPC blocked writing local run.py before any scientific execution; complete source was persisted under the remote TTIE release and published through GitHub API before the sole audit. Initial local tests were 2 pass / 1 fail (Spearman -0.9999999999999998 versus an exact -1 assertion). Only that synthetic-test assertion was corrected to tolerance 1e-14; server tests then passed. No scientific arithmetic or gate changed. Local worktree remains incomplete and is not the recovery source; GitHub and server home/F are authoritative.

Commands: python -m pytest research_log/T059F2/test_core.py -q -p no:cacheprovider --import-mode=importlib; python -m research_log.T059F2.run --e <frozen E artifacts> --f <frozen F result.json> --out <run>/artifacts/T059F2; python research_log/T059F2/verify.py <run>/artifacts/T059F2. Stop for research-lead review; no further method changes without authorization.
