# T069-B: BLOCKED by total-gradient consistency

The fixed endpoint gradient-cancellation diagnostic stopped during target-free development scoring because the three weighted float32 component gradients did not match the direct T062 total-objective gradient within the predeclared coordinatewise tolerance `abs(sum-direct) <= 2e-7 + 2e-5*abs(direct)`. This is the explicit BLOCKED condition in the authorization. No signal-present/absent conclusion is available.

Source: `3c4ce3b611918891254c71781daa80b547595429`; branch `codex/T069B-gradient-cancellation`. Task-owned files: `research_log/T069B/{authorization.md,PLAN.md,core.py,test_core.py,run.py,verify.py,binding.json,evaluation_binding.json,REPORT.md,evidence/*}`. Retain/review this task directory; the stacked historical branch is not a merge recommendation.

Implementation reuses exact CommonRegion2 and T062 losses, weights [1,10,5], frozen states and all 12 raw coordinates. Primary gradients share one float32 forward; detached gradients are promoted to float64 for cancellation scoring. The independent verifier reconstructs four separate forwards but was not reached. No tolerance, weights, coordinate subset, model, endpoint or optimizer was changed after the failure.

Tests: local affected suite `python -B -m pytest research_log/T069B research_log/T069A research_log/T067B -q`: 9 passed in 14.19 s. Same suite on A6000: 9 passed in 1.58 s. Synthetic aligned/opposed/zero cases, fixed weighted autograd and nearest-rank/strict threshold tests passed. Real endpoint consistency failed; independent verification NOT RUN.

Run: `20260921-125904-ttie-t069b-gradient-cancellation`, GPU 1 NVIDIA RTX A6000, source-pinned release `/media/wenchang/F/wjq/TTIE/releases/20260921-ttie-t069b-gradient-cancellation`; output `/media/wenchang/F/wjq/TTIE/runs/T069B-gradient-cancellation`. Command sequence after tests: `python -B -m research_log.T069B.run --out <output> && python -B -m research_log.T069B.verify --out <output>`. Full command is preserved in `evidence/run.sh`. Start 2026-09-21 12:59:08 +08; finish 12:59:13 +08; exit 1.

The failing endpoint's 12 absolute component-sum versus direct-total discrepancies, as printed in the original traceback, were:
```
[3.37604433e-08, 9.80217010e-08, 8.89122020e-09, 7.21774995e-09,
 1.11976988e-08, 4.22587618e-08, 1.31549314e-08, 5.19212335e-08,
 8.52858648e-07, 3.24100256e-07, 1.48374966e-07, 5.96046448e-08]
```
Maximum printed discrepancy: approximately 8.53e-7. The traceback did not record the image index or direct gradient values, so neither an exact violating-coordinate count nor its relative error is claimed. Different float32 backward reduction paths are a possible numerical cause, not an established explanation.

The failure occurred inside `endpoint_scores(False)` before the development freeze. Therefore no T99, transfer score freeze, joined labels, unsafe/safe flag counts or final diagnosis were produced. Transfer reference-quality reads in this execution: 0. Optimizer runs: 0. Model fits: 0. No fresh/final data or clean targets were accessed. The run was not retried with relaxed tolerance.

Source push initially hit a github.com:443 connection timeout; one unchanged retry succeeded. No email verification or authorization error occurred. Experiment execution itself was attempted once.

Recovery archive is verified locally and stored on both server filesystems; raw evidence and full source/run logs are preserved. Archives and SHA256 receipts are in `evidence/archives.json`.

Next step: research lead should review the failed predeclared numerical consistency check and authorize any narrowly scoped numerical investigation. Do not interpret this BLOCKED outcome as evidence for or against gradient cancellation, and do not build a guard or tune another statistic.

Evidence note: the wrapper metadata contains a stale historical releaseId; the explicit `cd`, run command and source-bound config identify the actual release above. Repository metadata is normalized to LF; original bytes remain in the recovery archive.
