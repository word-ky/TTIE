# T059-BR: functional cache compatibility established

All three frozen controls pass across 7,346 canonical rows. This establishes cache-path compatibility only; dual-tangent fitting remains untested and the T058-AF negative readiness finding is unchanged. No training followed the pass.

Exact tested source: `12bad36207ee2103540c65b97e7b57ac2e1e6dc9` (222 bindings). Authorization: `b7e5a3e5785773e4c59789d1d9f28e47c2f5435e`.

| Control / statistic | Actual | Expected | Gate evidence |
|---|---:|---:|---|
| Legacy/value x: value_huber | 0.051005665212869644 | 0.051005665212869644 | `{"actual": 0.051005665212869644, "expected": 0.051005665212869644, "abs_error": 0.0, "tolerance": 2e-06, "margin": 2e-06, "atol": 2e-06, "rtol": 2e-05, "passed": true}` |
| Legacy/value x: positive_fraction | 0.9976544976234436 | 0.9976544976234436 | `{"actual": 0.9976544976234436, "expected": 0.9976544976234436, "abs_error": 0.0, "tolerance": 1.9953089952468875e-05, "margin": 1.9953089952468875e-05, "atol": 2e-06, "rtol": 2e-05, "passed": true}` |
| Legacy/value x: median_cosine | 0.97242927551269531 | 0.97242927551269531 | `{"actual": 0.9724292755126953, "expected": 0.9724292755126953, "abs_error": 0.0, "tolerance": 1.944858551025391e-05, "margin": 1.944858551025391e-05, "atol": 2e-06, "rtol": 2e-05, "passed": true}` |
| Legacy/value x: direction_loss | 0.025412224233150482 | 0.025412224233150482 | `{"actual": 0.025412224233150482, "expected": 0.025412224233150482, "abs_error": 0.0, "tolerance": 2e-06, "margin": 2e-06, "atol": 2e-06, "rtol": 2e-05, "passed": true}` |
| Detail x: positive_fraction | 0.68456655740737915 | 0.68456653782440635 | `{"actual": 0.6845665574073792, "expected": 0.6845665378244064, "abs_error": 1.9582972798914966e-08, "tolerance": 1.3691331148147584e-05, "margin": 1.3671748175348669e-05, "atol": 2e-06, "rtol": 2e-05, "passed": true}` |
| Detail x: median_cosine | 0.23051643371582031 | 0.23051627609319664 | `{"actual": 0.2305164337158203, "expected": 0.23051627609319664, "abs_error": 1.5762262367546853e-07, "tolerance": 4.610328674316407e-06, "margin": 4.452706050640938e-06, "atol": 2e-06, "rtol": 2e-05, "passed": true}` |

Legacy eligible/ineligible: 7,248/98. Both detail controls: 7,244/102. Existing scalar math.isclose atol=2e-6, rtol=2e-5 unchanged.

Phi detail integrity: all 7,346 gradient rows pass frozen torch.allclose(2e-6,2e-5); max absolute error 1.6689300537109375e-6, max L2 error 1.992841399977046e-6, minimum nonzero cosine 0.9999999998472259. Accepted/reconstructed zero gradients: 98/98. Minimum element margin 9.66237394095515e-7. Phi detail positive fraction 0.6845665574073792, median cosine 0.23051634430885315 (alignment diagnostic only).

Feature discrepancy is diagnostic only: max absolute 4.050135612487793e-5, mean absolute 3.6406568524624293e-7; 7,042 exactly nonidentical rows, features 12–19 only. This differs from the prior 5,807 rows exceeding the extra element tolerance. Full quantiles, feature counts, means and maxima are in summary JSON.

Sole CPU run `20260917-202706-ttie-t059br-replay`, release `20260917-202644-ttie-t059br-replay`, exit 0, replay 28.100179849890992 seconds, torch 2.4.0+cu121. Local 4 tests passed in 7.52s; server 4 passed in 1.52s. Initial deploy SSH255 occurred before archive/upload; unchanged retry succeeded. No scientific rerun.

All asset/input/head before-after hashes agree. Training runs, optimizer steps, new source-image opens, reference-gradient recomputations, new feature/scorer forwards, target-domain/LOL-v2/official-test access all zero. No T059B fit.py or recipe edits.

Exact full receipt is gzip-compressed without changing its decompressed bytes; completion marker binds its original SHA256. Source bindings and all input/output tensor hashes are inside. Raw archive is verified on home and F and through local tar readback. Local extraction hit ENOSPC and left a partial receipt; that partial file is not authoritative. GitHub publication uses the verified archive in memory. No historical artifacts deleted.

Raw archive SHA256: `6d15404ef9ce465d4d92b2cf51d567c73f4edf79a4e3cbdd711cac42da9b809a`. Stop for research-lead review; no training or self-merge.
