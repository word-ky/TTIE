# T059-B — preflight stopped; fit feasibility unresolved

**PARTIAL. No training ran and no new checkpoint exists.** The sole execution stopped at `run.py:56`, `CACHED_FEATURE_PATH_MISMATCH`, before the requested frozen-checkpoint source-fit statistics or optimizer creation. This is not a `dual-tangent source fit not feasible under fixed recipe` result: that hypothesis remains untested.

## What stopped

The implementation added a cached-feature elementwise equality check `allclose(atol=1e-6, rtol=1e-6)` between the stored T014 training features and T059-A cached phi. That threshold comes from the existing T014 derivative-record feature check, but applying it across these two accepted caches is an **additional implementation precheck**, not the task's explicitly requested aggregate frozen-checkpoint baseline replay. It stopped too early to determine whether those required baseline statistics would reproduce. No tolerance was relaxed and no setting or code was changed to rerun after failure.

A separate read-only comparison of existing hash-verified cached tensors, with autograd disabled and no model forward or training, localized the difference:

| Quantity | Observation |
|---|---:|
| Compared tensor shape | 7346x28 |
| Rows failing implementation feature check | 5807 |
| Elements failing check | 12941 |
| Differing feature indices | 12–19 only (CLIP score features) |
| Other20 feature rows | Exact equality |
| First mismatch | row0, feature13 |
| First stored/cached values | -1.5427707433700562 / -1.5427734851837158 |
| First absolute difference | 2.7418136596679688e-06 |
| Maximum difference | 4.050135612487793e-05 (row970, feature14) |
| Mismatched active/inactive rows | 5738 / 69 |

The raw tensors were not changed. The numerical cause is not adjudicated; these deltas alone do not establish failed gradient/fit statistics. The diagnostic preserves all per-feature counts/maxima, first/max values, row identity, input hashes, and before/after byte equality. It did not evaluate the missing frozen-head statistics or train a model.

## Fixed implementation and test evidence

Authorization `a2041ad4095a54b7d63086c898498b565c1c6ed8`; tested source `2cecef0797e625440e474720adcea6b8195d1447` with217 exact Git-blob/source bindings. Branch `codex/T059B-dual-source-fit`; PR #97. Reuses `train_head` unchanged: seed7, original initialization, train-only normalization, CPU, AdamW1e-3/1e-4, same shuffled batch order,100epochs, final-epoch checkpoint. Only intended loss change is adding the detail cosine term with total weights1:1:1. The original recipe explicitly fixes CPU, so changing training to GPU would violate this task's fixed recipe; cached T059-A Jacobians were already generated on A6000.

Local4tests15.65s/server4tests2.70sPASS. With detail supervision disabled, the new wrapper reproduces original T014 training parameters and value histories bit-for-bit. These synthetic unit tests are not the requested source-training experiment.

Sole run `20260917-185145-ttie-t059b-dualfit`, release `20260917-185120-ttie-t059b-dualfit`, exit1; failed preflight 15.556082566967234s, ended 2026-09-17T10:53:47.577660+00:00. Exact wrapper/log/error/failure receipt are attached. The first startup SSH255 timeout happened after meta.json and before run.sh/start. Read-only inspection confirmed no process,tmux,log,or scientific artifacts. We completed the wrapper from that persisted command and launched the same run ID once. `submission_recovery.json` records this; no scientific rerun occurred.

## Missing results and boundaries

No required frozen-checkpoint replay table, epoch loss history, trained head hash, new fit statistics, or feasibility classification exists. Do not treat this as a negative dual-fit result or a successful preflight. The run read authorized cached source feature/value/legacy/detail supervision, but opened no new source image and recomputed no reference gradient. No optimizer update, target-domain/LOL-v2/official-test access, inference/API change, or deployment occurred. The failure receipt explicitly records `training_started:false`; the post-training full before/after-state receipt was not reached.

Source217 hashes were verified unchanged during archiving. Existing T014-bank and T059-A-cache bytes used in the read-only diagnostic were verified before/after. Raw failure archive `/home/wenchang/asdasdsad/wjq/TTIE/shared/t059b/T059B_evidence.tar.gz` and `/media/wenchang/F/wjq/TTIE/shared/t059b/T059B_evidence.tar.gz`, 1791bytes, SHA256 `f5d2721c4f7d03b38fc8b0f3e63a86080271f72e605b327110e60fa82ad1fece`, verifiedhome/F/local with every archive member checked. Post-stop feature-difference evidence is separate from the immutable original failed-run archive.

Next: research-lead review of the implementation-added early gate and cached score-feature differences. Decide in a separately bounded task whether/how to perform the originally requested frozen-head statistic replay. No training, tolerance change, or rescue rerun is undertaken in this cycle.
