# T062-A — NEGATIVE by fixed four-gate contract

The fixed zero-reference objective raises mean development PSNR substantially but fails the worst-tail and RGB-SSIM gates. Exact classification: **the fixed three-term zero-reference objective is insufficient**. No alternative objective, weight, target or step budget was tried.

| Measure | Observed | Gate | Pass |
|---|---:|---:|---|
| Mean PSNR delta vs T036 | +3.718047528776 dB | >= +0.50 | Yes |
| Regressions vs exact T026-A | 12/100 | <=29 | Yes |
| Worst PSNR delta vs T026-A | -7.105521504373 dB | >= -5.614 | No |
| Mean RGB-SSIM delta vs T036 | -0.009366946841 | >= -0.001 | No |

Absolute T062-A PSNR/RGB-SSIM: **14.948087665788 / 0.336935369812**. Median PSNR delta vs T036: **+3.860070371005 dB**; improve/regress/tie **85/15/0**. Selected steps: `{"40": 88, "28": 1, "32": 3, "35": 2, "31": 1, "30": 1, "34": 1, "24": 1, "36": 1, "37": 1}`. These are development-only measurements. The PSNR gain supports investigating the objective as a bottleneck, but does not satisfy the fixed safety/quality contract or promote this control.

Tested executable source: `734bf4b00af139bc69644f7c870489a1af5a26df`; authorization `dda0be7ec8f182aeea702c1ba0bb9727a196089a`. Release `20260920-ttie-t062a-zr`; A6000 GPU1 run `20260920-041332-ttie-t062a-zr`. Unchanged T036 CommonRegion2/CommonBox, low-only gate, identity 12 raw coordinates, Adam0.03,40 updates. Lspa uses the mean over all concatenated horizontal/vertical RGB first-difference errors after nonoverlapping4x4 pooling; Lexp16x16,target0.60; Lcol global RGB means; weights1/10/5. Only the fixed objective supplies gradients and selects the minimum-loss/earliest-tie state. CLIP is used solely for the unchanged initial gate; no T014 learned energy or T059/T060 method invocation.

All 100 trajectories (4,100 rendered images plus raw states/losses/output hashes) were fsynced before global freeze **2026-09-19T20:15:53.699084+00:00**; freeze SHA **`301c041aedeb7d4925560a860417891e6bc64fa616da3c1c10565b10d3496d00`**. First reference/baseline read **2026-09-19T20:16:08.773953+00:00**. Every low/cohort/code/asset binding is in `binding.json`, `evidence/config.json`, and the freeze. Inference has a low-only data-open guard after model/code bootstrap; evaluation is a separate process. No clean/reference/PSNR/SSIM/baseline-outcome quality field entered inference or selection. No official/cross-dataset access.

Verification: 100 selected outputs independently recomputed with NumPy pooling/differences/color arithmetic (no main loss function): max objective error **5.9556344100819558e-07**. Earliest minimum-loss selection replay agrees. Independent PSNR/dot-product and RGB-SSIM checks: max metric error **9.0949470177292824e-13**; scalar aggregate replay error **8.8817841970012523e-16**, identical gates. Main inference file hashes were checked before references were opened. Detailed100 checks and paired table are committed.

Tests: 5 original action tests passed locally21.63s;3 new loss/gradient/trajectory tests passed locally22.68s and remotely2.31s. Initial baseline collection needed the existing tests directory added to PYTHONPATH. Canonical Git-byte deployment corrected98 Windows line-ending hash differences before execution; all129 deployed bindings matched. No numerical/code behavior was changed by normalization. Harmless dependency and scalar-conversion warnings are in logs.

A runtime cleanup issue occurred **after** `ALL_100_FROZEN`: the still-active read guard blocked Python temporary-directory cleanup (`/tmp/tmpmwoa_3__`), producing recursion tracebacks in finalization. The denied operation was directory cleanup, not a reference or outcome read. The separate evaluator completed, all evidence checks passed and the job exited0. This incidental cleanup problem is recorded without altering frozen source or rerunning the experiment. Full log retained.

Inference127.845629s, evaluation28.033670s. Bulk output directory `/media/wenchang/F/wjq/TTIE/runs/T062A-fixed-zr`. Full raw archive12,387,092,480bytes on F (one physical archive copy); recovery1,196,312bytes verified locally and on home/F. `archives.json` contains exact paths/SHA-256. Recovery includes all traces, receipts, code and logs; full images/selected tensors reside in the raw archive. Commands are preserved in `evidence/run/run.sh`.

Next: stop this exact three-term objective pending research-lead review; do not sweep weights/exposure or relax failed gates. Official and cross-dataset tests remain sealed.
