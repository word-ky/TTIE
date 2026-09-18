# T059-K — DONE / REFERENCE_ORACLE_ONLY

Source `e9a4485e1b4a20864f92a84b662bcb099d50a922`; authorization `b78d99ed78e56390b112ba450aa97fd93a4b4d6c`.

marginal scalar support exists in the fixed inner-train pool; the unresolved failure is localization/conditioning of that support rather than absence of scalar values.

Fixed 48-image / 4,357-row training pool; 1,529 held queries. No source-pool enlargement, training, or rollout. Train-LOO excludes the query image. Huber delta=1 float32; exact ties choose the smallest canonical global row ID. A6000 physical GPU1 computes oracle losses and float64 squared feature distances; CPU independently recomputes every choice, loss, tie, count, range, five-donor membership and distance rank. Percentile=(rank-1)/(candidate_count-1); squared distances preserve Euclidean ordering.

| Metric | Train-LOO | Inner-held |
|---|---:|---:|
| global_oracle_huber | 0.0003226476546842605 | 0.0005118412664160132 |
| target_in_global_scalar_range | 0.9977048427817305 | 0.9967298888162197 |
| frozen_five_membership | 0.006885471654808354 | 0.007848266841072597 |

Fixed gate 0.07650849781930447. Margins: {"train": 0.0761858501646202, "heldout": 0.07599665655288845}. All descriptive distributions and per-query evidence are in result.json and oracle_rows.pt.

Candidate table persisted and SHA-bound before held scalar access in a separate process. Initial accepted G/I/J recorded metrics and hashes checked before construction; exact rowwise scalar replays occur after freeze to preserve the accepted information boundary. Only training scalar storage is read before freeze; the mixed scalar file full hash is deferred until evaluation. Input/source hashes remain unchanged. No reference-gradient tensor deserialization. All prohibited-access and training counters are zero.

Tests: 2 passed in 1.43s. Sole scientific run `20260918-105152-ttie-t059k-global`, 02:51:57–02:52:18 UTC, exit 0. Independent CPU verifier PASS for all 5,886 queries; maximum GPU/CPU distance difference 2.842170943040401e-14, exact ranks. No scientific failure or rerun. Nonblocking NVML warning. Local D: ENOSPC prevented fetch/local artifact copies; source and evidence persisted under the remote project root and GitHub through memory streaming.

Commands: `python -m pytest research_log/T059K/test_core.py -q -p no:cacheprovider --import-mode=importlib`; `python -m research_log.T059K.run freeze --out "$AUTODL_RUN_DIR/artifacts/T059K"`; `python -m research_log.T059K.run evaluate --out "$AUTODL_RUN_DIR/artifacts/T059K"`; `python research_log/T059K/verify.py "$AUTODL_RUN_DIR/artifacts/T059K"`. Environment: CUDA_VISIBLE_DEVICES=1, OMP/MKL/OPENBLAS threads=1.

Stop for research-lead review. This source-target oracle establishes marginal scalar support, not a deployable selector or benchmark improvement.
