# T020-C analysis

**Development OOF negative: 3/5 clauses pass.** The unchanged representation and learner do not satisfy the predeclared non-spatial in-domain OOF safety contract. This does not establish that no predictor can use these features. It does not support proceeding directly to combined-domain final training. No fresh-safety or deployment claim.

| Group | H1/H0 | H1/H* | Beneficial/equal/harmful | No move/x only/y only/both | x/y/joint agreement |
|---|---:|---:|---|---|---|
| nonspatial_pool | 0.947906080161726 | 1.054803218551833 | 37 / 63 / 20 | 62 / 11 / 13 / 34 | 85 / 89 / 73 |
| clean | 1.2072654157145626 | 1.6730731697808663 | 0 / 39 / 1 | 39 / 0 / 1 / 0 | 38 / 38 / 38 |
| homogeneous_dark | 0.9370790223162285 | 1.0356040364068448 | 21 / 15 / 4 | 15 / 4 / 7 / 14 | 33 / 33 / 29 |
| homogeneous_bright | 0.9701882237639385 | 1.0946176124638984 | 16 / 9 / 15 | 8 / 7 / 5 / 20 | 14 / 18 / 6 |

Literal acceptance vector: `[true, false, true, true, false]` (pooled, clean, dark, bright mean-MSE safety; clean zero harmful). Clean makes one y-lower move; all other clean predictions stay at center. Exact per-axis class counts and per-condition diagnostics are in `T020C_run/summary.json`. No post-result change or search.

Exactly the accepted T020-B 40 image IDs /120 episodes; historical fold bytes SHA256 `8fdb03cdac63af5d6e58b557c96679bc15c1e2e921e525916eb9d21ad22cd2c1`. Five folds each train on32 image IDs/96 rows and hold out8 image IDs/24 rows. Reused byte-identical `direction_probe.py`, `direction_probe_run.train_fold`, `deadband_probe.training_targets`; same seed7,100 epochs, final epoch,84→64→64→3 SiLU, unweighted CE, AdamW1e-3/wd1e-4,batch256. Training-only population normalizers are persisted per head. Ten experiment heads total; no final head.

GPU1 A6000 computed600 candidate feature vectors once from cached opaque input pixels, cached canonical selected state and cached gate. Reused unchanged `fresh_deadband.select.cross_vectors`/`energy_model.features`, frozen CLIP and prototypes. No reference pixels or reference-MSE decoding during extraction, no TTT rerun. CPU 2.13.0+cpu training preserves the accepted donor recipe/runtime; GPU Torch 2.4.0+cu121. Five cross vectors are scattered into donor9-slot layout; only the same five slots enter unchanged84-D construction.

Feature SHA256 `df191f8b8b9019931b9500cb40b2ada7b01e8b35ca0f15c6bb4d8e876fae63b8`. Prediction SHA256 `731258100e6a71ec439935a06349c7c40851e35842bee0d72df3a45de67e01e8`. Predictions frozen 2026-09-13T12:43:39.265794+00:00; independent reference-free replay completed 2026-09-13T12:44:03.281439+00:00; held-out evaluation opened 2026-09-13T12:44:16.805424+00:00. Training extracts only bx/by for fold training rows from accepted target bytes, without decoding held-out targets or reference-MSE values. Semantic IDs only assign groups, never model inputs. `hard_index` preserves donor27-grid convention (3×nine-grid index); evaluation uses `score_index` in the nine-hard table.

Baseline tests3 passed(15.30s); affected tests5 passed(8.40s), including exact84-D scatter and both-axis synthetic held-out-label poison tests comparing prediction hashes after unchanged training. Independent verifier imports no TTIE: reconstructs features, normalizers,10 head logits/classes/boundaries before reference evaluation, then all120 MSE joins, diagnostics and five literal clauses. Both stages PASS. Supplemental accepted-B cache binding replay:120 rows/360 cached input-state-gate files exact.

Source commit: `a4f9893c4486bd7ade6138e5f44047d29e9285c0`. GPU run: `20260913-204217-ttie-t020c-features`, exit0. One pre-launch transport issue: CRLF in generated Git index text; converted index metadata to LF and repeated source preflight successfully. Source and scientific settings were unchanged; no failed scientific run.

Stopped after this single fixed experiment. No T020-A fresh features/logits/references or per-row outcomes were opened; no heterogeneous training, second seed, threshold search, confidence gate, fresh cohort or T020-D. Await research-lead review.
