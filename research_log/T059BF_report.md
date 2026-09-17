# T059-BF — dual-tangent source fit feasible

All five fixed source-fit gates passed after exactly one 100-epoch fit (2,900 optimizer steps). This is source-training feasibility, not generalization, real-domain performance, or a deployable rollout. No further run followed the verdict.

## Fixed comparison

| Statistic | Frozen T014 | Final head | Required | Margin |
|---|---:|---:|---:|---:|
| detail_positive | 0.68456655740737915 | 0.98799008131027222 | >= 0.75 | 0.23799008131027222 |
| detail_cosine | 0.23051643371582031 | 0.6266331672668457 | >= 0.5 | 0.1266331672668457 |
| legacy_positive | 0.9976544976234436 | 0.99282562732696533 | >= 0.94999999999999996 | 0.042825627326965376 |
| legacy_cosine | 0.97242927551269531 | 0.96652674674987793 | >= 0.90000000000000002 | 0.066526746749877907 |
| value_huber | 0.051005665212869644 | 0.072232343256473541 | <= 0.076508497819304466 | 0.004276154562830925 |

All 7,346 canonical rows retained; legacy eligible/ineligible 7,248/98, detail 7,244/102. Value Huber increased, but remains within its preregistered upper bound. Full old/new direction loss and row counts are in summary JSON.

## Preflight and fixed recipe

Before optimizer creation, the original aggregate preflight passed: all four legacy/value statistics exactly match T014; detail positive-fraction error 1.9582972798914966e-8 and median-cosine error 1.5762262367546853e-7 pass unchanged math.isclose(atol=2e-6, rtol=2e-5). Exact actual/expected/tolerance/margin table is in verification JSON. The diagnostic cached-feature maximum delta remains 4.050135612487793e-5.

The only scientific edit deletes the invalid elementwise x/phi assertion. fit.py is byte-identical to PR97. Features remain stored x; original EnergyHead, normalization, seed7 CPU initialization, AdamW, batch256/order, 100epochs, final-epoch checkpoint, and loss weights1:1:1 are unchanged.

## Execution and evidence

Authorization `10d61751324976496e027a0339f37a3a5e9828b1`; exact tested source `524f6436a2282492d69e32cfb3ab67b37b750979` with218 Gitblob/SHA bindings. Sole run `20260917-211326-ttie-t059bf-fixedfit`, release `20260917-211304-ttie-t059bf-fixedfit`, exit0; elapsed runner 41.476537461043335 seconds. Focused local tests2 passed22.73s; repeated directory invocation2 passed16.21s; server2 passed2.54s. Repeating local tests did not run the source-bank experiment.

Command: `python -m pytest research_log/T059B -q -p no:cacheprovider --import-mode=importlib && python research_log/T059B/run.py --out "$AUTODL_ARTIFACTS_DIR/T059BF"` under fixed one-thread CPU settings.

Saved checkpoint SHA256 `f736df30cf31d0f45974242cd74d2bbcf00a0dc2b920a3852c435b8a0538b8fe`; reopened after execution. All100 epoch losses saved, history SHA256 `3d2586d98fc6e111dd47ebada6c94208547f493a15f88c3f0f951b9046abbd06`. Initial/final head tensor hashes retained in receipt/verification; immutable assets and input tensors compare equal before/after, and all assets were rehashed during read-only postrun verification. Full exact receipt is gzip-compressed; its decompressed bytes match the completion marker. No new feature forward, source-image open, reference-gradient recomputation, target/LOL-v2/official-test access or inference-reference leakage.

Raw archive 260152 bytes, SHA256 `2f253b562c7f1340f02c7faf36a5b747f7a1c1e44b011176ee47fac44e36f930`, verified remote home/F. Local compact fetch stopped during history transfer due D-disk exhaustion; partial local downloads are not authoritative. The exact server export was read into memory and hash-checked for GitHub API delivery. A post-completion process grep returned exit1 because no matching training process remained; train.log explicitly records exit0. No scientific failure, rerun, rescue or setting change.

Stop for research-lead review. No real-domain evaluation or self-merge.
