# T061-C — NEGATIVE: fixed source step 11 does not transfer sufficiently

The one immutable candidate fails all five preregistered gates on the fixed 100-image development cohort. Exact classification: **a single source-chosen fixed stopping step does not transfer sufficiently**. Stop the fixed-global-step route pending lead review; no other step was tried.

| Gate | Observed | Required | Pass |
|---|---:|---:|---|
| Mean PSNR delta vs T036 | -2.251547746536 dB | >= +0.20 | No |
| Median PSNR delta vs T036 | -1.650304240129 dB | > 0 | No |
| Regressions vs T026-A | 92/100 | <= 29 | No |
| Worst PSNR delta vs T026-A | -7.610363062768 dB | >= -5.614 | No |
| Mean RGB-SSIM delta vs T036 | -0.070266757319 | >= -0.001 | No |

PSNR improve/regress/tie: **10/90/0** vs T036; **8/92/0** vs T026-A. Mean PSNR delta vs T026-A is -1.312290678849 dB. The full paired table is `result/paired.json`; absolute means including identity/no-adaptation are in `absolute_means.json`. These are development diagnostics, not final baseline-gap claims.

Authorization: `9bece47c33da0465faf5165e2f9bca74fb183c32`. Tested source `ac345e834e1fddd505558cdad54b2334a7b9f954`. T061-B candidate manifest SHA `34a15a13f0c94b07a7eef870efee9f28da51e30abc7f440a342a62da5c0cd1b5` fixes k*=11. T037-A accepted CSV commit `c0d84b1d3c7e6af186c28ca736d6ac2bc752d35c`; exact Git blobs and SHA-256s are in `result/intent.json`.

Evaluation-intent freeze: **2026-09-19T19:45:33.196210+00:00**; SHA **`941f8c356c9db0adeb64d5a0fc05f2c291a333e601691640d109cf93e982ded5`**. First development-quality-read marker: **2026-09-19T19:45:33.511795+00:00**, recorded immediately before CSV parsing in the separate evaluation process. Preparation only hashed the CSV bytes, without parsing or displaying quality fields. Prior T061-A historical summary exposure is disclosed; it did not change the fixed T061-B candidate or the original five gates.

Validation: 100 unique image identities and exact table order; all 4,100 common states present, uniquely ordered 0..40 and finite. Original T036 and exact T026-A selected PSNR/SSIM reproduce their respective per-step entries exactly. Two synthetic tests passed in 0.110 s, covering fixed-step lookup/arithmetic and missing, duplicate, nonfinite, order and selected-output mismatch rejection. Independent verifier repeats all 100 fixed lookups and 80-digit Decimal aggregate arithmetic; maximum checked aggregate error is **0.0**, identical gate booleans and classification. Prepare, evaluate and verify each ran once successfully; no rerun or deviation.

No new optimizer, adaptation, rendering, source re-selection, image/reference decoding, T059-E invocation, official-test or cross-dataset access. CPU table arithmetic only. Commands and script hashes are in `commands.json`, and each phase has a saved log. Reproduction uses the bound input bytes and a fresh output directory; run prepare before evaluate, then verify.

The immutable step chosen on source data degrades this development cohort substantially. This result rejects the tested single global step; it does not justify another step, a new per-image selector, or a revised gate without a new research-lead task.
