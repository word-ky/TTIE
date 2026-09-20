# T065-B — DONE / TRANSFER_NEGATIVE

Source commit: `6f01920f580b7c8e1f816eb96df236c3f58d1e7c`. Branch: `codex/T065B-balanced-safety`. Authorization: `d442a47e7a4f97d56f20507581f7b69573e36583`.

The exact class-balanced linear safety guard fails transfer. Despite identifying80/81 unsafe development states, it triggers0/100 development and0/100 transfer rollbacks. Both prior catastrophic transfer states are predicted safe at the clipped-logit ceiling probability0.9999999999999065, so their unsafe base choices remain unchanged. This is an exposed-cohort audit, not fresh qualification. Close this exact head; no threshold/weight/lambda/feature retry.

Transfer absolute PSNR/SSIM: 15.471845204573 / 0.397896864536.

| Gate | Required | Development | Transfer | Transfer pass |
|---|---:|---:|---:|---|
| mean_delta_psnr | >=2.00 dB | 3.5695322981528443 | 3.8619303358477532 | True |
| median_delta_psnr | >0 dB | 3.093839109589621 | 3.814406659510415 | True |
| regressions_t026 | <=29/100 | 3 | 12 | True |
| worst_delta_t026 | >=-5.614 dB | -4.074446413712419 | -10.364944679494553 | False |
| mean_delta_ssim | >=-0.001 | 0.015011775217397242 | 0.01842184098158023 | True |

Training counts, recorded before fitting: safe2719, unsafe81, total2800. Safe weight 0.5148951820522251; unsafe weight 17.28395061728395; each class has total weight1400. Labels use development-only PSNR(state)-PSNR(T026)>=-5.614. The fixed0.5 rule passes allfive development gates, permitting transfer.

Development state confusion matrix:

| True class | Predicted safe | Predicted unsafe |
|---|---:|---:|
| Safe | 2503 | 216 |
| Unsafe | 1 | 80 |

Unsafe->safe false negatives:1/81; safe->unsafe:216/2719. These are state-level development counts, not transfer qualification.

Development selected-step histogram (nonzero): {"21": 1, "22": 1, "23": 2, "24": 1, "25": 9, "26": 21, "27": 65}.
Transfer selected-step histogram (nonzero): {"18": 1, "21": 3, "24": 3, "25": 6, "26": 22, "27": 65}.

Post-freeze tail outcomes:

| Index | Step | Selected PSNR / SSIM | Margin vs T026 | Predicted safe probability |
|---|---:|---|---:|---:|
| 16 | 21 | 15.243922691784 / 0.730831917479 | -7.131115430332 | 0.9999999999999065 |
| 86 | 25 | 13.247833801770 / 0.607845717517 | -10.364944679495 | 0.9999999999999065 |

Exact reuse: `research_log/T065A/core.py` feature code SHA256 `fec15f6f2cb430377c1c52e2316a1036a1c6878a16ec37bdf113293017a0487f`; original T065-A model/normalization SHA256 `5a644c74b7b98a242e7c2945f2fa7b971a5d0e046ba3e66f4e5e7462ba7599ef`. The11feature definitions and all means/scales are reused unchanged. No new normalization fit, feature or cohort. Renderer, low inputs, frozen states and rho=0.9857470621423519 remain fixed. Development baseline anchors are accepted Git-bound values; all development state reference metrics are independently recomputed.

Solver: one float64 CPU Newton/IRLS fit, zero initialization, summed class-weighted logistic NLL +.001*||w||^2/2, unregularized intercept. Logits clipped[-30,30], fixed50-iteration maximum, early stop only step infinity norm<1e-12. The run reaches the50-update cap: final step infinity norm 8.38220372764547e-12; it does not meet the stricter early-stop threshold. No extra iterations or second model were attempted. Full before/after coefficients, steps and loss are in `evidence/newton_trace.json`. Independent equation residual max 7.511324895403959e-12, coefficient difference 7.841549631848466e-11. This capped result is the requested solver result, not a claimed early-convergence event.

Model SHA256 `a36d1bdf46cbe9cdb38074c88024df615ad8bca5fb7e23d46ff666b714699819`. Model/normalization/class-count/training/source/state manifest frozen 2026-09-20T11:31:42.981745+00:00; all100transfer decisions/outputs frozen 2026-09-20T11:32:06.697760+00:00; first transfer reference-quality read 2026-09-20T11:32:06.769307+00:00. Selector SHA256 `b23d2400b76d9d286a2e4e9582076028ef7cd60b83b2f9d0f520b843fe229d20`; transfer freeze SHA256 `9665b0dde9fb7bad6d644a96f502894fe994f8702a113121d2878ec2bc817f95`. Target selection reads only100low PNGs+100frozen traces and the frozen model. Transfer table contains target-free features/probabilities and hash metadata, no reference outcomes. Tail metrics are produced only after the freeze. No transfer labels, oracle steps/ranges, official test or cross-dataset data entered training/selection.

Independent verifier PASS: re-render2,800states per cohort; independently compute features, labels, class weights, Newton fit and every saved equation/update/stop condition; reproduce normalization, probabilities, rollback choices, hashes, read ordering, confusion matrix, gates and classification. Development/transfer feature max errors 7.105427357601002e-15 / 6.217248937900877e-15; metric max errors 1.0871303857129533e-12 / 8.846257060213247e-13.

Validation: 11baseline passed16.01s;4classifier passed8.32s;5including independent passed8.17s;16affected passed8.49s; remote16passed1.55s. No failed tests, source revisions after outcomes, remote errors or scientific retries. Source pushed before run `20260920-193108-ttie-t065b-safety`; exit0, primary pipeline 50.996854s. A6000 GPU1 handles renderer/primary features; CPU handles the authorized float64 Newton solve/probabilities and independent numerical audit. Adam reruns0; exactly one primary logistic fit plus required independent recomputation.

Commands: `python -B -m pytest research_log/T065B/test_core.py research_log/T065A/test_core.py research_log/T063C/test_core.py research_log/T063A/test_core.py -q`; `python -B -m research_log.T065B.run --out /media/wenchang/F/wjq/TTIE/runs/T065B-balanced-safety`; `python -B -m research_log.T065B.verify --out /media/wenchang/F/wjq/TTIE/runs/T065B-balanced-safety`. Single-thread BLAS, sequential MKL, TF32off and CUDA_VISIBLE_DEVICES=1 as in source/run receipts.

Artifacts: `research_log/T065B/evidence/` includes full feature/label table, counts, Newton trace, model, normalization binding, development confusion/gates/histogram, transfer probabilities/freeze, post-freeze metrics/tails, verification and logs. Raw `/media/wenchang/F/wjq/TTIE/shared/t065b/T065B_raw.tar`, SHA256 `9c0aa38ec8e5d6fe7ee13e71736ed600f1d553efd7908a1770a0f00619248882`, 293744640 bytes. Recovery `/home/wenchang/asdasdsad/wjq/TTIE/shared/t065b/T065B_recovery.tar.gz`, also F: and local, SHA256 `0b8c5678734d0dd7c8a6e1991f5457ff1d05fa408a9822c27e31f57bf57f0437`, 2240528 bytes. Home/F/local recovery hashes match.

Recommendation: accept TRANSFER_NEGATIVE and close this exact class-balanced linear guard. Good development unsafe-state recall does not translate into detection of these late transfer failures. Await a new bounded task; this result alone does not establish that every possible model over the11features must fail. No historical artifacts were merged into main; only this completion mailbox entry is written there.
