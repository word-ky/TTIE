# T059-T: frozen first-order/finite-step audit — negative

**optimizer-scale/curvature mismatch not established; close the direct detail-step branch**.

Authorization `2d1fae2970c7ea8dd4564d2028b8e30ca7988faf`; source `8871ba34df1743cf30647dc0b4ee2c35aac36fa3`; exact parent S evidence `27d2c2d41c6a2aebfd1b16ab78b9bf2e0ad853e3`. No new action, renderer, optimizer, model/feature forward or clean image access. CPU-only algebra over persisted tensors.

## Fixed result

Exact80 unique state0 banks/16 images aligned, with the same61 inherited reference-gradient-eligible anchors. **57/61=0.9344262295081968** have L<0, passing90%. Of the4 actual harmful anchors, **0/4** are overshoot flips (L<0,A>0), failing50%. Median Adam saturation fraction **1.0** passes80%. Therefore the second scientific stop condition applies. Neither T059-S nor its gates are reclassified.

The four actual harms already have positive first-order true-MSE change along the persisted Adam displacement. This does not support the specified explanation of a locally descending step crossing into finite-step harm. It does not establish broader impossibility of detail adaptation or authorize alternative actions. Raw-gradient alignment and alignment along Adam's coordinatewise-rescaled displacement are distinct quantities.

L=<g_R,v1>, A=mse1-mse0, R=A-L, all in absolute MSE units. No baseline-MSE ratios in gates. Fresh Adam replay is algebraic, not an optimizer execution: lr.05, betas.9/.999, epsilon1e-8. Maximum replay error **1.2359145491747103e-8**, below inherited1e-7 tolerance. Saturation denominator contains only abs(g_hat)>eps coordinates; all eligible denominators nonempty. Spearman uses average exact-tie ranks; quantiles use linear interpolation.

| Quantity | p10 | median | p90 |
|---|---:|---:|---:|
| L | -0.00012595167472100497 | -4.4333987526732836e-05 | -4.2794797471496722e-06 |
| A | -0.00012026918114620838 | -4.4265280127527007e-05 | -4.0199547608199548e-06 |
| R | -6.2221096818181787e-07 | 8.7327299069849072e-07 | 5.6824935747965845e-06 |
| predicted_gradient_norm | 0.031453661388567547 | 0.10678693438564199 | 0.25181609253867626 |
| reference_gradient_norm | 9.4859579981332933e-05 | 0.00026729817542185842 | 0.00059882981208929267 |
| step_norm | 0.2465385903822008 | 0.36054978608512178 | 0.3999984418107973 |
| saturation_fraction | 0.98181818181818181 | 1 | 1 |

Spearman(predicted gradient norm,step norm) **0.3504494976203067**; Spearman(reference gradient norm,step norm) **0.642199894235854**. Descriptive only.

## Four eligible harms

| Bank / image / global row | L | A | R | Saturation |
|---|---:|---:|---:|---:|
| 84 / 38829 / 1418 | 4.9460784893435618e-06 | 5.1615316082574692e-06 | 2.1545311891390733e-07 | 1 |
| 131 / 39956 / 2224 | 3.2949329453523411e-06 | 3.6455370178503643e-06 | 3.5060407249802322e-07 | 1 |
| 284 / 45229 / 5137 | 9.4125576854694469e-06 | 9.7550316812203275e-06 | 3.4247399575088063e-07 | 0.97826086956521741 |
| 305 / 46031 / 5572 | 1.5763467927883195e-12 | 2.5971345947508405e-06 | 2.5971330184040476e-06 | 0.92000000000000004 |

Complete61-row diagnostics, all80 compact input vectors/IDs/eligibility, all four harmful-anchor receipts, counts/distributions/correlations and machine-readable classification are committed.

## Evidence and execution

All31 published T059-S files verified against exact Git blob identities, then SHA256-bound. All82 frozen action files/field/decisions plus source reference-gradient/evaluation records checked against original S freeze and published copies. Inputs immutable before/after, no replacement or recomputation. Full80/61 alignment and reference eligibility replayed; canonical IDs and each g_hat match the original field. Input hashes and7 own source bindings are in input_receipt.json.

Focused **3 tests pass1.54s**: dot/residual/overshoot algebra, epsilon-restricted saturation, ordered exact gate boundaries. Sole run `20260919-003356-ttie-t059t-forensic`, **2026-09-18 16:34:01–16:34:13UTC**, exit0. Independent implementation uses scalar-loop dot/norm and explicit first-moment/second-moment bias corrections, pairwise average ranks, scalar quantiles and separately written gates. **All80/61 alignment, all61 diagnostics, summaries and classification PASS.** No scientific failure, code repair or rerun. One authorization-transfer SSH timeout recovered before execution. LocalD full; server project and GitHub are authoritative.

Counters all0: new_actions,renderer_calls,optimizer_steps,model_or_feature_forwards,outer_supervision_reads,target_domain_access,lolv2_access,official_test_access,inference_reference_leakage. No reference images opened, no GPU workload needed for this small persisted-value audit.

Commands: `pytest -p no:cacheprovider --import-mode=importlib research_log/T059T/test_core.py -q`; separate `python research_log/T059T/run.py --out OUT` then `python research_log/T059T/verify.py --out OUT`, with PYTHONPATH and single CPU threading. Source committed before execution. Raw and recovery archive paths/hashes in archives.json/recovery.json, both home/F verified. Existing large S tensors remain in their accepted archive and are not duplicated.

Stop and await research-lead review. No magnitude-preserving action, optimizer variant, second step, sweep, new cohort, C2 outer, target-domain rollout or self-merge.
