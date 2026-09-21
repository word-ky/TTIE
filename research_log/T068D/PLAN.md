# T068-D fixed component-regret diagnosis

Reuse T068C source dc9f766c16ad25971bb1a36a52eedc0a035733c2, its endpoint/data/label/rank/report flow, and T062A exact losses. Prior baseline local10passed1skip15.49s, remote11passed1.98s, independent4015GPU states PASS. Internal same-repo code, no new dependencies/external code.

New logic only float64 component weighting [1,10,5], interval min/max, sum-regret/sum-excursion, prescribed singleton/zero handling. Full source components/history bound by accepted T066A data. Recheck against frozen trace components. Source already includes exact accepted renderer/low-only losses. Keep probabilities/endpoints unchanged.

Increment1 focused extraction/min/max/ratio/degeneracy/nearest-rank tests. Increment2 reuse C orchestration, separate labels post-freeze, affected tests, source commit/push and full real-data independent verification. Primary uses frozen components, no optimization. Verifier rerenders every interval state on A6000 GPU1 and directly recalculates T062A loss components from low image and rendered output, requires exact component reproduction, then uses independent Python min/max/sums for score. T99/flags/labels/diagnosis verified independently with atol1e-12 for score arithmetic. Exact threshold rank sorted_R[98], strict >, no post-result adjustment.

Development100score/T99 freeze before transfer; transfer100score/weighted endpoint/min/max/a/e/fullZ/state/output/rule hashes before reading or hashing label evidence. Ranks descending1+strictly greater count; percentiles fraction<=score. Primary T067C endpoint margins independently crosschecked T066B labels. No clean/reference images opened, no hypothetical guard evaluation. No selectors, newcohorts, fitting or tuning. Stop after one diagnostic report.
