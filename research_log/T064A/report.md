# T064-A — FRESH_TAIL_SELECTION_LIMITED

Both T063-D safety failures have safe states in their already-frozen prefix. **100/100 images have a nonempty safety-reachable set**, and the fixed safety-constrained reference oracle passes all five gates. Formal diagnosis: **FRESH_TAIL_SELECTION_LIMITED**, strictly **REFERENCE_ORACLE_ONLY**. The two failure images are selected beyond their safe interval by the frozen normalized-progress rule. This does not establish a deployable way to find the safe checkpoints and does not reverse T063-D's negative qualification verdict.

Authorization main `b6c8730e306205465f960e34435e4473b5e36892`; source `b82b47ccc47cdc3bc9494e4a80b378888db33430`; branch `codex/T064A-fresh-tail-reachability`. Inputs are the exact accepted T063-D cohort/freeze/states/low images/controls. Cohort SHA3206ea57061f4b45164a81f105f818de6d6d15b342a77797ce9f1eaaccc52554; accepted input freeze SHA7cdb658d380720f51c2519817953a9e0b6dc0d6f4f2b4bca1deb0bb5cdb3cdd2.

| Safety-constrained oracle measure | Result | Requirement |
|---|---:|---:|
| Mean PSNR delta vs T036 | +5.452321463119 | >=+2.00 |
| Median PSNR delta vs T036 | +4.891642318974 | >0 |
| Regressions vs T026 | 0/100 | <=29/100 |
| Worst paired PSNR delta vs T026 | +0.427951960342 | >=-5.614 |
| Mean RGB-SSIM delta vs T036 | +0.037089770304 | >=-0.001 |

All gates PASS. Oracle absolute17.062236331845dB /0.416564793858RGB-SSIM. Unreachable images: none. Oracle choices:65at27,35earlier; full histogram in evidence/summary.json. No global step was selected or promoted.

| T063-D failure | Frozen rule choice | Safe prefix steps | Reference oracle | PSNR selected /oracle /T026 |
|---|---:|---|---:|---|
| index16,low00478.png | 21 | **8..19** | 14 | 15.24392269 /28.61900528 /22.37503812 |
| index86,low00262.png | 25 | **8..19** | 14 | 13.24783380 /30.13017242 /23.61277848 |

For index16, selected/oracle/T026 RGB-SSIM=.730831917479/.845094865168/.827508093042. For index86: .607845717517/.822459445778/.790280790804. Both safe sets are exactly the contiguous interval8..19. All28PSNR/SSIM/safety values for each failure are in evidence/tail_trajectories.csv (56rows) and tail_failures.json; all2,800state metrics are in per_step.csv/per_image.json. These oracle steps/ranges/harm indicators must never become deployable per-image inputs.

Reconstruction uses the frozen standalone low PNG, saved active mask/12-D state and accepted CommonRegion2 renderer only. It checks low file/tensor hashes, trace hashes, accepted source/cohort/config binding and all100currently-selected output tensor hashes. No historical output container is loaded during reconstruction. All2,800renders freeze at **2026-09-20T06:40:14.936437+00:00**, SHA `b5da79c68ecd1b05488e7b22f6b5b58e46e4a7c4ed2ed1627b5291f80f77da21`; first diagnostic reference/quality marker **2026-09-20T06:40:16.801380+00:00**, strictly after freeze. Source bindings258; data-read audit records exactly the allowed100low files plus100traces. After freeze, evaluator additionally verifies torch.equal between every reconstructed chosen checkpoint and accepted selected output and reproduces all accepted control/current-choice metrics.

The oracle is unchanged: reachable iffPSNR(state)-PSNR(T026)>=-5.614; select maximum reference PSNR within reachable,earliest-step tie. An empty set would be flagged unreachable and the unconstrained maximum shown only diagnostically. Classification is selection-limited only when all100reachable and allfivegates pass. No alternative oracle or threshold was tried.

Independent verifier PASS: separately GPU-re-renders all2,800states, checks every hash and100selected-output matches, source bindings and low/state-only reads, freeze/reference ordering, and independently recomputes dot-product PSNR/separable-convolution RGB-SSIM, reachable sets, earliest-tie choices, aggregate gates and classification. Maximum metric discrepancy **1.1759482276829658e-12**; verification UTC **2026-09-20T06:41:47.196176+00:00**. All accepted selected-output equality and control metric checks pass.

Tests:3baseline pass13.93s;3focused pass9.26s;final3pass14.26s local /1.51s remote. Coverage:nonfinal selected-state reconstruction/identity,safe-set earliest tie,unreachable reporting. Sole run `20260920-143921-ttie-t064a-prefix`, release `20260920-ttie-t064a-prefix`, exits0. A6000GPU1 for primary/independent rendering;8CPUworkers for metrics. Reconstruction 43.749874s;primary metrics 32.013535s. No Adam/optimizer runs, source mismatch, reconstruction failure,scientific rerun,rho change,selector fit,new cohort,official test/cross-dataset access or lead-file edit.

Archiving encountered a missing home backup directory after both F archives were already created. The directory was created and only the pending copy/hash/receipt steps were completed; scientific outputs and archives were not regenerated. Recovery verified local/home/F. Full raw images remain `/media/wenchang/F/wjq/TTIE/runs/T064A-fresh-tail-oracle`; raw archive `/media/wenchang/F/wjq/TTIE/shared/t064a/T064A_raw.tar`,8065914880bytes,SHA `e01a41b40fd92ec820bdbe0890df4e7a6240ecd6fb63b491e6596aab6ed04289`. Recovery755645bytes,SHA `c02528edb7393fe363ce215739e5401119afd2edc6125de2c2f89e93b2250f4a`;paths in archives.json. Recovery includes source/config/100receipts/allper-state diagnostics and runlog; image tensors are in the raw archive. Original T063-D inputs remain archived separately. Summary/report are derived afterward from immutable outputs.

Next: research-lead review of the diagnosed selection bottleneck. Stop; no new target-free statistic,selector,oracle-informed fallback or additional dataset is authorized in this task.
