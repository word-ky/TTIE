# T059-R — DONE

2026-09-18T12:13:41.404694+00:00

**Classification:** `full-curve cross-head disagreement is not a convincing tail-uncertainty mechanism; stop the dual-head disagreement line`.

Authorization `c855b29275a203b6de2c110c5d576ca038db0a7c`; source `39e5c40f0400bedb65bd56b61418ab45d29aa11d`; branch `codex/T059R-rank-disagreement`. Sole run `20260918-201204-ttie-t059r-rank`, release `20260918-ttie-t059r-rank`, 12:12:08–12:12:13 UTC, exit0. CPU-only existing-value audit; no model or feature forwards.

**3 unsafe / 63 non-singleton banks**, 17 singletons excluded. **AUROC = 0.8222222222222222**, below 0.90. Spearman(u, positive E harm) = **0.23401366800652135**.

| Unsafe bank / image | rho_EM | u | Descending rank / 63 | Percentile from most uncertain | Harm |
|---|---|---|---|---|---|
| 230 / 43581 | 0.21739130434782608 | 0.782608695652174 | 3 | 3.225806451612903 | 4.667080879211426 |
| 280 / 45229 | 0.9294117647058824 | 0.07058823529411762 | 31 | 48.38709677419355 | 6.364080429077148 |
| 305 / 46031 | 0.5339130434782609 | 0.46608695652173915 | 4 | 4.838709677419355 | 6.284384727478027 |

| Group | Count | Median u | Q25 | Q75 | IQR width |
|---|---|---|---|---|---|
| unsafe | 3 | 0.46608695652173915 | 0.2683375959079284 | 0.6243478260869566 | 0.3560102301790282 |
| safe | 60 | 0.06701484623541887 | 0.036346752219890005 | 0.1158830871645517 | 0.0795363349446617 |

All three conditions fail: AUROC <0.90; bank280 lies outside the top uncertainty quartile; the maximum-harm bank280 also lies outside the top decile. Its high cross-head rank concordance (rho=0.9294117647058824) leaves it at rank31 rather than marking it uncertain. Do not rescue this line with post-hoc thresholds/margins/top-k rules.

**Frozen conventions:** average ranks for exact score ties; rho is Pearson correlation of those ranks; u=1-rho; descending u then smallest bank ID. One-based rank is converted to percentile `100*(rank-1)/(N-1)` from most uncertain, fixed in source before descriptors/targets. Top quartile uses percentile <=25 (ranks1–16 of63); top decile <=10 (ranks1–7). Argmin ties use state_index then canonical global ID. No constant non-singleton curve was encountered. AUROC uses exact pairwise comparisons with half credit for tied u; summaries use linear quartiles. No new gate/policy is defined.

Exact E inner-held split: **16 source images / 1529 rows / 80 banks**. Reused accepted Q canonical global/bank/state/image alignment and normalization metadata, checked against the accepted E split and M access manifest; E/M p and anchors are extracted again from their original persisted storage ranges, without models or target-tensor decoding. Complete per-bank scores, row IDs, descriptors and ranks are in descriptors.json; target-attached full63-bank table is evaluation_table.json. All 80 E harm values, including zero singleton harms, exactly reproduce Q.

Pinned E source `830e80ba0e1a4d09bce9c9ffd57be162a781d013`, head SHA256 `e15e91c4e9be401bcac6d3de039ec2d1ae7141fb6cec40c388d22549652d43d0`, prediction artifact `976d8b68e4988623c954230a87577acf5e4d46b5c4583e35bb530fdf1a37b61d`. M source `09a599b5f5da7dc61a5e20716a4ffcf470b0bcda`, head `0146c300d6f7ab067a1760da1f459722d0d6c9a77a3d1d256b60f446e2fd28da`, prediction artifact `a9a4a10aa644b393095fbca3a68b4dd09328548821bdf099ffe15404f7539dda`. The 319 input/source bindings and own source hashes are unchanged before/after.

Descriptors frozen **2026-09-18T12:12:10.265405+00:00** before target read **2026-09-18T12:12:11.841140+00:00**. Full-file hashes bind mixed prediction/target artifact bytes, while selective ZIP storage reads decode only requested prediction/metadata tensors before freeze. Source delta_t is reopened only in the separate evaluate process. All requested counters zero: training_runs, optimizer_steps, model_forwards, new_feature_forwards, source_target_reads_before_descriptor_freeze, outer_supervision_reads, target_domain_access, lolv2_access, official_test_access, inference_reference_leakage.

**Validation:** 4 focused tests passed in 1.33s: average tied ranks, undefined constant statistic, singleton exclusion and deterministic rank ties, AUROC/quartiles/gate conditions, and target replacement invariance. Independent verifier recomputes average ranks by pairwise counting (not the implementation's sorted groups), Spearman, uncertainty order, argmins, all metrics and classification. AUROC is independently replayed through the Mann–Whitney rank-sum identity. All63 descriptors and all80 Q E harms pass exact/tight numeric checks. No failures, repairs or scientific reruns this cycle.

Commands: project-venv pytest with `-p no:cacheprovider --import-mode=importlib`; separate `python -m research_log.T059R.run describe`, `evaluate`, then `python -m research_log.T059R.verify`. Files under research_log/T059R include implementation/storage/tests/verifier, authorization and source/input bindings, descriptor and target/evaluation artifacts, report and log. D: remains full; remote project persistence and authenticated local API publication used, with no server credentials.

Result SHA256 `d78497dc512a9b220193d6ea05f9897119d8c0073c3b681398d879d57a5ad1a0`; descriptors `d6174e2c70f5500318558bff37b354d4e6a50644cc74903495e9f4e62f9416dd`; prediction rows `24ba8f9bcaa8123e3ffaf087b61a67190c748d6673ba5c6acf3a66b83d3bb89b`; target rows `7ddafc6fd1424a60cb428a39cafc4ab04f0b655a3693342ef4a0c9bf1d20a959`; evaluation table `567c8b554f3bd7767c8805125314e2b422734fcb5c8bd2f69e60a597ebe2f52d`. Raw/recovery evidence is retained on both remote project disks with matching hashes.

**Next:** stop the dual-head-disagreement line and await research-lead review. No new gate/threshold, C2 outer, target-domain/LOL-v2, official test, training, rollout, deployment or self-merge is authorized. This diagnostic is already-opened source evidence, not fresh validation.
