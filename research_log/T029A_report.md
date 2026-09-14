# T029-A: frozen-field alignment diagnosis

Status: DONE diagnostic, with execution deviations disclosed below. Overall predeclared classification: **weak/mixed field alignment**. This task does not promote or change a deployable method.

All 4,100 accepted states have nondegenerate active-coordinate gradients. The learned field is often aligned early, but commonly opposed to reference MSE late in the trajectory and at the already-frozen selected state. Positive dot means that the raw-space direction `-g_E` is first-order descending for reference MSE; this is a local diagnostic, not a quality improvement from an executed update.

| Frozen subset | Count | Mean cosine | Median cosine | p10 | p90 | Positive dot |
|---|---:|---:|---:|---:|---:|---:|
| All steps | 4100 | 0.165587296 | 0.171865263 | -0.521997294 | 0.805371860 | 58.3414634% |
| Existing selected steps | 100 | -0.201663905 | -0.278469368 | -0.568391348 | 0.245862074 | 24% |
| Step 0 | 100 | 0.437520272 | 0.694133810 | -0.730970042 | 0.840306184 | 80% |
| Step 10 | 100 | 0.665495828 | 0.726601211 | 0.346426993 | 0.879095239 | 97% |
| Step 20 | 100 | 0.123456179 | 0.210545900 | -0.437137214 | 0.539657384 | 67% |
| Step 30 | 100 | -0.238039820 | -0.220573222 | -0.597480271 | 0.172935645 | 22% |
| Step 40 | 100 | -0.192109640 | -0.245285485 | -0.560552298 | 0.176585964 | 24% |

Energy-zero, reference-zero and either-zero fractions are all 0, including every step and selected subset (`norm<=1e-12`). Inactive identity coordinates are excluded. Every image has 41 states and 2 times its active-region count coordinates; all 41 step summaries are in `T029A_result/evidence/summary.json`. No new state was selected using reference information.

## Frozen provenance and information boundary

- Accepted T026-A source `b2359721c89db732d17e03be273e0bdb71bb377a`, evidence `d577fc24a54cdb0e3de22bc3d07dd14168c10e70`, merge `51f84a9d96880bca8e908c9b3cdff496d7277e39`, run `20260914-113853-ttie-t026a-gamma05`.
- Split SHA256 `b88c8347005984b5523b117b52c0c068672fe172eb7c9aa5b60102d350e2d85b`. Accepted freeze/config/metrics hashes and 300 per-image artifact hashes are in `T029A_preflight/preflight.json`.
- T014 energy SHA256 `c3d1eef9f20af163e1268823cf16d3fdaed315e7723fc94db3b1138ad1336521`; CLIP, prototype and gate hashes are bound in the same preflight. All assets were rechecked after the audit.
- Fourteen transitive accepted source blobs are byte-bound in `T029A_alignment/source_binding.json`. The unchanged path is `gamma_range_ttt.evaluate_energy -> SemanticScorer/FrozenCLIP -> energy_model.features (T013-28-v1) -> frozen T014 EnergyHead`. Gate, Region2, gamma lower0.5/EV box, float32 renderer/features, seed7 and TF32-off are unchanged. Reference RGB MSE accumulates in float64, as in the preceding reference diagnostic; both gradients are float32 with float64 active-vector reductions.
- Preflight `20260914-172718-ttie-t029a-preflight` completed **09:27:53.169590 UTC**, SHA256 `b382c60b1fb78cb50c88b362b8af504dc108cd29d9f302f174b6a06641d36f8f`. It freezes all 4,100 raw-state and reconstructed-output hashes; every grid and all 100 selected outputs reproduce exactly. Normal images decoded: zero.
- Task reference deployment began **09:28:23.150197 UTC**, after that freeze. Only the fixed100 named validation references were copied into `shared/t029a/REFERENCE_GRADIENT_DIAGNOSTIC_ONLY/normal`. Existing references from earlier tasks remain outside this audit's explicit decode allowlist. Official-test names/content were not enumerated or decoded.
- `reference_gradient.py` contains only the quarantined reference-MSE derivative. No optimizer exists in the audit. Raw tensors are restored from frozen states, unchanged during differentiation, and `.grad` remains unset; head/scorer parameter and buffer hashes remain unchanged. There are **zero optimizer updates and zero reference-driven selection decisions**. No oracle/baseline/training or test execution.

## Execution and verification

Complete GPU audit source `da4ce46d376a77c9b41217724e4304787f4dc807`, release `20260914-173746-ttie-t029a-diagnostic`, run **`20260914-173750-ttie-t029a-alignment`**. It completed4100 states at09:43:32.202214 UTC in333.684406563s; per-image mean3.239222579s. A6000 physicalGPU1, Python3.12.12, Torch2.4.0+cu121/CUDA12.1, seed7, float32, TF32off, accepted `deterministic_algorithms=False`. NVML warning did not block actual CUDA.

- Local focused tests:2 passed23.82s; remote:2 passed1.53s. Tests cover active-vector exclusion, classification boundaries, no raw mutation and a finite-difference reference derivative.
- Every restored raw state/output/gradient is finite; reconstructed output hashes match preflight at all4100states. Original feature difference: exactly0. Maximum absolute difference from4000 historical CUDA energy gradients: `5.245208740234375e-6`, reported as auxiliary evidence rather than hidden or used for a new gate.
- Thirty predeclared independent samples (image indices0,10,...90 x steps0,20,40) use fresh raw leaves, functional renderer calls and separate backward passes. Maximum gradient difference `6.556510925292969e-7`; reference gradients match exactly. Maximum sample norm/dot/cosine differences: `2.869461886989555e-7` / `2.227334545273907e-8` / `1.5965176736187914e-6`. Float32 sample check uses rtol1e-4/atol1e-6.
- Independent NumPy reconstruction from saved full gradients reproduces all4100 scalar records and43 summaries (all, selected,41steps), rtol/atol1e-12; maxscalar difference `2.220446049250313e-16`. The same CPU check passes on Windows and server. No new GPU/image computation was used for this verification repair.
- Artifact archive, receipt file hashes, original preflight bytes and deployed source hashes verified locally.

## Failures and deviations — not a single process attempt

There is one complete4100-state audit, **plus two aborted GPU attempts**. This deviates from the literal single-run instruction and is explicitly submitted for research-lead review. No settings or scientific outputs were selected across attempts.

1. `20260914-172838-ttie-t029a-alignment` stopped during the first image on an unnecessarily strict historical-gradient equality check: absolute difference2.0116567611694336e-7.
2. `20260914-173030-ttie-t029a-alignment` completed95images before the same extra check stopped on1.3634562492370605e-6. I removed this unrequested historical equality gate rather than keep widening it. Complete image progress is now saved to recover from the observed failure. Neither failed attempt produced a final summary; their logs/source/commands remain archived.
3. Final GPU computation completed, but its wrapper exited1 when the CPU verifier compared independently recomputed float32-derived scalars at an inconsistent1e-9 absolute tolerance. One failing sample had cosine difference6.00e-7. Postprocessor source `9f4e53d2ae593dce1f931cc194f0c97ff1a69514` uses the same float32 tolerance as the gradient check; exact saved-gradient NumPy reductions remain1e-12. CPU-only recheck exited0 and produced PASS. The original exit1 log is preserved; it is not described as an exit0 GPU wrapper.
4. Local D: filled during evidence download. Sparse checkout removed duplicate tracked working copies, preserving task artifacts, outer project logs and Git history. Remote computation was uninterrupted; no driver changes.

## Evidence and next action

`T029A_result/evidence/` contains full per-state rows, all gradient vectors, deterministic independent samples, step/selected summaries, runtime/immutability receipt and independent check. `T029A_preflight/` contains all bound raw states, hashes and reference-deployment receipt. `T029A_result/runs/` contains all four execution logs/commands/metadata, including preflight and failures.

Full source, progress and execution backup on F: `shared/t029a/T029A_execution.tar`,3911680bytes, SHA256 `9c3b9f6747d6392201a66208144a67bfd213db1ed5e67ac9dcc5a43cf9cd39da`. Compact archive SHA256 `9bda693c6bd2a8145c2ce2aacf682483b45e31058afa72ba86e1290e96118c95`,935418bytes, verified in both server roots and locally. `T029A_backup.json` binds archives and deployed files.

PR54: https://github.com/word-ky/TTIE/pull/54. Stop here for research-lead review of the late-trajectory mismatch and execution deviations; no retraining, recalibration, new stopping rule, bound change or promotion was performed. Scientific state remains research-lead owned.

weak/mixed field alignment
