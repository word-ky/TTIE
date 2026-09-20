# T063-A — DONE / SELECTION_HEADROOM_PRESENT

**REFERENCE_ORACLE_ONLY.** All100 images have a safety-reachable saved T062 prefix state; the fixed reference-dependent oracle passes all five unchanged aggregate gates. This establishes selection headroom within the existing trajectory, not a deployable stopping rule. No optimizer, gate/objective rerun, fitted selector or new cohort.

Authorization `b42367491a9bb3fc3c2f5286bf47328beac09548`; tested source `c4c58706c23c9c8972e2720eae87d294d5163986`; branch `codex/T063A-prefix-reachability`. Inputs are accepted T062-C-R2 evidence33d7fdb0/source47bcab6b, global freeze700ae2612234eb139a20c3560b58c85dca1eef3849347c24ffc09575620c07de and immutable cohort e8a66a350bcce5282df2e8114355afc434bcd0f4ea69e54957ebb7618f9183c2.

| Diagnostic gate | Oracle value | Requirement | Result |
|---|---:|---:|---|
| Images with a safe state | 100/100 | 100/100 | PASS |
| Mean PSNR delta vsT036 | 5.222190307682 dB | >=2.00 | PASS |
| Median PSNR delta vsT036 | 4.235285911349 dB | >0 | PASS |
| Regressions vsT026 | 0/100 | <=29 | PASS |
| Worst PSNR delta vsT026 | 0.875735240405 dB | >=-5.614 | PASS |
| Mean RGB-SSIM delta vsT036 | 0.029564754312 | >=-.001 | PASS |

Oracle absolute mean **17.359724060428dB /0.432159288494 RGB-SSIM**. There are no SAFETY_UNREACHABLE images.62oracle choices remain at27;38choose an earlier state. No global step was selected, ranked or promoted.

The sole fixed-step27 safety failure is index66, Train/Low/low00221.png. Its safety-reachable set is exactly **12..24**. Reference oracle step17 yields **32.353762650854dB /0.765392886880 SSIM**; accepted T026 gives22.832859726919dB /0.792368584916SSIM, whereas T062step27 gives15.498684380821dB /0.636919547360SSIM. Full28-row failure trajectory is in step27_failure_trajectories.csv; all2800 rows in per_step.csv. These per-image steps and harm labels are diagnostic only and forbidden as inference inputs/selectors.

Reconstruction directly copies each saved12-D state into the exact accepted CommonRegion2, uses its saved active mask and frozen low tensor, and renders sequentially on A6000GPU1. All100 reconstructed step27 outputs are **bit-exact** to accepted T062-C-R2 output tensors. All2800images freeze at **2026-09-20T02:19:45.141847+00:00**, SHA **87ce16ac06dc421ed5293c34205b3de6ad128341f57e3bb7ad9e8c2c53e497ca**; first diagnostic reference/quality marker **2026-09-20T02:19:46.969683+00:00**, strictly after freeze. No normal/reference image is read by reconstruction. Source/config233bindings, original output/trace hashes, per-state hashes, cohort identity and timestamps are persisted.

The separate offline evaluator opens only the same exposed100 normals. For each image the immutable oracle maximizes PSNR among states meeting deltaT026>=-5.614, earliesttie; if empty it would report maxPSNR and mark unreachable. No objective, weight, step budget, action, asset or threshold changes. Existing T026/T036/identity and step27 metrics reproduce from exact frozen outputs.

Independent verifier separately re-renders **all2800** from the original frozen low/states on GPU and reproduces every hash; rechecks original step27 equality and read ordering; recomputes all28-state metrics using dot-product PSNR/separable-convolution SSIM; independently reconstructs reachable sets, earliesttie oracle, scalar aggregate gates and classification. Maximum primary-versus-independent state-metric difference **1.04449782156735e-12**. Verification PASS. This does not establish that any target-free selector can obtain the oracle result.

Validation:3 focused tests pass13.41s locally /1.54s remotely (frozen-state renderer identity/equality, earliesttie, unreachable reporting). Syntax checks pass. Sole run `20260920-101902-ttie-t063a-prefix`, release `20260920-ttie-t063a-prefix`, exits0. Reconstruction35.583397s; primary offline metrics32.290997s with8CPU workers. GPU1 is used for both original reconstruction and independent reconstruction; no Adam/optimization runs. Source committed/pushed before scientific execution. No failures, reconstruction mismatch, reruns, new fresh data, official-test/cross-dataset access, heuristic fitting or lead-file edits.

Raw reconstructed tensors: F/runs/T063A-prefix-oracle; full archive `/media/wenchang/F/wjq/TTIE/shared/t063a/T063A_raw.tar`,8065546240bytes SHA `ae8cd61c007d6f1399ed66f78bb8c77c85821b88741ca03796ce9fffc0958e51`. Recovery620929bytes SHA `28e0fb59df8ccd75f04dc140df586485084708dfedfcc095c199ec3384328f33`, verified local/home/F; contains source/config, receipts, complete per-state metrics/reachable sets and runlogs. Original T062-C-R2 frozen inputs remain separately archived.

Next: research-lead review of selection headroom. Stop here. Any future degraded-image-only stopping/selection mechanism requires a separate task; never reuse the reference oracle as a deployable selector or infer held-out efficacy.


## PR review follow-up — test collection

Addressed review comment4055819038: configure pytest importlib mode and repository/test helper paths by default, and qualify four legacy test-module imports that relied on path insertion or ambiguous top-level names. No algorithm, fixture expectation, frozen source binding, metric, or experiment output changed.

The reported two-test basename collision was reproduced before the fix. Final affected run without PYTHONPATH and without an explicit import-mode flag passes26tests plus10subtests in26.08s. Full repository collection now collects382tests with no duplicate-module/import-name errors, but is **not fully green**:8existing errors remain (7share a historical pinned-source hash assertion;1vendored baseline needs cv2). Both root causes reproduce with the previous prepend import mode and the new config overridden. Logs are committed; these unrelated failures are left unchanged. No GPU/scientific rerun.
