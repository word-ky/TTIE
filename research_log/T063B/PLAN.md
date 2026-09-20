# T063-B fixed loss-balance stopping audit

Authorization: main 68b1948a (full SHA in source binding and local state); exact task in authorization.md.

Reuse accepted T062A.losses, CommonRegion2, frozen T062-A development traces and T062-C-R2 transfer traces. Reproduce every rendered state hash and recompute the three components on GPU. Use binary64 ratio arithmetic with denominator floor 1e-8. Candidates are unique finite development ratios plus numpy/Python nextafter sentinels; largest admissible step, identity if none. Offline calibration reuses the already independently verified T062-B development metric rows, bound by canonical Git hashes and matched state hashes. No transfer quality/artifact reads before global selected-output freeze.

Baseline: T062A and T063A focused tests, 6 passed in 26.09s. Increment 1: ratio/selector/grid/calibration unit tests. Increment 2: GPU reconstruction, calibration manifest, guarded target-free transfer and offline evaluation. Independent verifier recomputes components and ratios on GPU, all candidate choices and scores, threshold tie-break, transfer selected states and hashes, and independent reference metrics. It runs only after primary freeze/evaluation, and cannot feed choices back to the primary selector.

Full affected regression means the reusable loss/renderer tests plus the new rule tests, followed by end-to-end server verification. Existing unrelated full-repository collection errors remain unchanged. No optimizer, new cohort, held-out access or second statistic.
