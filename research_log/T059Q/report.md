# T059-Q — DONE

2026-09-18T10:37:30.373408+00:00

**Classification:** `exact dual-head agreement is not a sufficient target-free safety gate; stop`.

Authorization `7a1f070ef1a25c13b40820a47f9797b203585c9d`. Frozen-decision source `8473e9a55e0c6ec271775731ad1127ca5ff1c860`; reporting-repair source `4be3ca6ed25c0c8db6b8e61765e6d2178209920c`; branch `codex/T059Q-consensus`.

| Policy | Oracle-hit fraction | Mean regret | Median | p90 | Maximum | Harmed vs state0 |
|---|---|---|---|---|---|---|
| E | 58/80 (0.725) | 0.24020987916737796 | 0.0 | 0.08642201870679866 | 6.364080429077148 | 3/80 (0.0375) |
| M | 52/80 (0.65) | 0.2690812815912068 | 0.0 | 0.1460764527320867 | 7.348230361938477 | 3/80 (0.0375) |
| Q | 52/80 (0.65) | 0.21365644820034504 | 0.0 | 0.5538098931312562 | 6.364080429077148 | 1/80 (0.0125) |

Non-singleton exact agreement **41/63 = 0.6507936507936508**; 17 singleton banks excluded from this denominator and select state0. Coverage passes >=0.50. Compared with E, Q lowers mean regret and harm count, but **does not lower maximum regret and worsens p90**. Thus the preregistered comparative gate fails.

Known F2 boundary banks: 230 (E state13, M state1) and 305 (E state21, M state1) abstain to state0 and remove harm; **bank 280 / image 45229** has both heads choose **state14 / canonical row5078**, retaining regret and harm **6.364080429077148**. Exact agreement can preserve a shared catastrophic error. The full prior-tail table covers the union of F2's stored worst10-by-limit-Huber, top10-by-original-regret and all boundary banks (14 distinct banks), and is descriptive only.

Exact accepted E inner-held cohort: **16 images / 1529 rows / 80 banks**. E row global/bank/state/anchor tensors match the canonical split. M's accepted heldout_access.global_indices, bank/image IDs and anchor tensor establish identical order; the accepted M load_side code constructs this order. E/M normalization metadata match exactly. No checkpoint/model forwards were performed. Prediction argmins use unchanged persisted `p`; ties use state_index then canonical global row. Policy is unchanged exact agreement or state0, with no score calibration or secondary rule.

Pinned E source `830e80ba0e1a4d09bce9c9ffd57be162a781d013`, head SHA256 `e15e91c4e9be401bcac6d3de039ec2d1ae7141fb6cec40c388d22549652d43d0`, prediction artifact `976d8b68e4988623c954230a87577acf5e4d46b5c4583e35bb530fdf1a37b61d`. M source `09a599b5f5da7dc61a5e20716a4ffcf470b0bcda`, head `0146c300d6f7ab067a1760da1f459722d0d6c9a77a3d1d256b60f446e2fd28da`, prediction artifact `a9a4a10aa644b393095fbca3a68b4dd09328548821bdf099ffe15404f7539dda`. All input/source hashes remain unchanged.

Predictions and targets are co-stored in accepted .pt artifacts. Full-file hashes bind bytes; the adapted existing ZIP storage reader decodes only requested prediction/metadata tensors before freezing decisions. Its byte ranges and hashes are recorded. No target tensor was decoded before freeze **2026-09-18T10:31:48.849577+00:00**; first target access **2026-09-18T10:31:50.346147+00:00**. The reporting recovery reused all three prediction/decision/freeze files byte-identically, then reevaluated at **2026-09-18T10:35:26.993113+00:00**. E's persisted delta_t is the sole evaluation target. Differences use the persisted FP32 coordinate, summaries float64, p90 linear interpolation; oracle hit means selected target equals the minimum, including ties.

**Observed reporting failure and minimal repair:** run `20260918-183141-ttie-t059q-consensus` (10:31:47–10:31:50 UTC) froze decisions, then exited 1 at an overly strict assertion equating current delta_t-based regret to historical F2 t-based regret. All 80 selected states were identical; only subtraction roundoff differed, at most **5.960464477539063e-08** across 7 banks. The repair checks selected-state identity and reports that difference. Original source, failed log and artifacts remain intact. Run `20260918-183520-ttie-t059q-report-recovery` (10:35:25–10:35:28 UTC) reused frozen decisions and completed evaluation plus independent verification, exit0. No model/decision rerun or scientific recipe change.

Tests: **4 passed in 1.35s**, and **4 passed in 1.43s** after the reporting-only repair. Focused tests cover state/global tie order, disagreement abstention, singleton exclusion, target replacement invariance in mixed artifacts, summaries and gate precedence. Independent replay verifies every 1529 prediction/target row and all 80 decisions, exact ties, agreement, abstention, regret/harm, comparative conditions, classification and F2 tail table. All required counters are zero: training_runs, optimizer_steps, model_forwards, new_feature_forwards, source_target_reads_before_decision_freeze, outer_supervision_reads, target_domain_access, lolv2_access, official_test_access, inference_reference_leakage.

Commands: project-venv pytest with `-p no:cacheprovider --import-mode=importlib`; separate `python -m research_log.T059Q.run decide`, `evaluate`, and `python -m research_log.T059Q.verify` processes. CPU was used for this small persisted-value audit; no GPU workload exists. Files include core/run/storage/tests/verifier, authorization and source/input bindings, full decision/evaluation tables, raw prediction/target rows, freeze marker, result, verification and logs. Local D: remains full, so project artifacts are persisted remotely and published via local authenticated GitHub API without server credentials. Two initial inspection filenames were corrected (E summary/complete, M run.py); no scientific computation occurred during those failed lookups.

Result SHA256 `e4f34e27a982cb73507442b0964cfa3c1422b4baff101b967f63b5d63843b6ea`; decisions `ecf60165378abaa008e53cd394ac693a7ed5fd8a0cfc8cc6b38b767aff561973`; prediction rows `24ba8f9bcaa8123e3ffaf087b61a67190c748d6673ba5c6acf3a66b83d3bb89b`; evaluation table `c62f2c1ca72a5b2574beedc5a52c1ec2c4cc7dfe6764dfddcb34944379c31781`; target rows `e5dc28583a8417185ae8e66e0fb4ebd076cfbc4e1791c64ab166bcf7a2cd6258`. Raw evidence and recovery will be hash-matched on both remote project disks.

**Next:** stop and await research-lead review. This is the already-opened source diagnostic, not fresh validation; no further rule, training, C2 outer, target-domain/LOL-v2, official test, rollout or deployment is authorized.
