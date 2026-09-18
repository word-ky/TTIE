# T059-M — DONE

removing joint Sobolev losses does not rescue unseen-image scalar transfer; scalar conditioning/generalization remains unsupported under the current 28-D EnergyHead.

Source `09a599b5f5da7dc61a5e20716a4ffcf470b0bcda`; authorization `507a06254a3e25f6671ab568211549a5ec34b577`.

| Metric | Inner-train | Inner-held |
|---|---:|---:|
| huber | 0.005251884460449219 | 0.2274731993675232 |
| margin | 0.07125661335885525 | -0.15096470154821873 |
| baseline | 0.05690297484397888 | 0.22107574343681335 |
| change | -0.05165109038352966 | 0.006397455930709839 |

Fixed gate 0.07650849781930447. Train fits well, but held fails. Removing joint Sobolev losses alone does not rescue transfer under this fixed recipe; this does not establish a universal impossibility for the representation. No alternate fit attempted.

Unchanged EnergyHead 28→64→64→1 with SiLU, seed7 initialization, CPU float32, AdamW lr0.001 / weight_decay0.0001 / betas(0.9,0.999) / eps1e-8, batch256, exactly100epochs and1,800steps, final-epoch checkpoint. Population-std normalization uses training raw features and log(MSE+1e-6), identical to accepted E buffers; initial per-tensor head hashes exactly match E. Feature and MSE tensors reproduce accepted E hashes. The sole scientific change is scalar-only bank-relative Huber; accepted_E_fit.py and value_loss.diff document the exact change. No legacy/detail tensors or objectives.

Only 4,357 training rows from48images/240banks are read for fitting; held1,529rows from16images/80banks are opened separately. Unique state0 anchors match accepted hashes. C2 outer remains unopened. All original source bindings and split/input hashes replay. Raw frozen features are fed through the unchanged head standardization to preserve E arithmetic exactly. No model/feature/optimizer changes.

Checkpoint, optimizer, generator state, epoch history, initial/final hashes, train predictions/metrics/access receipts all persisted and fsynced before marker 2026-09-18T04:31:37.620825+00:00; held scalar access begins 2026-09-18T04:31:39.337373+00:00. Head SHA256 0146c300d6f7ab067a1760da1f459722d0d6c9a77a3d1d256b60f446e2fd28da; optimizer SHA256 b9e963c821c4376526ee190b07f2a0e90ce025ad66841203bc48286d8b114eea; history SHA256 aef1bb448868da17934a5520afef7523a6b5d798255514eb47a2896f864d2692. Before/after inputs and frozen outputs unchanged.

Tests2 passed in1.66s; accepted baseline/source replay PASS before fitting. Sole fit run20260918-123128-ttie-t059m-scalar, 04:31:32–04:31:41 UTC, exit0. Independent frozen-checkpoint evaluator reproduces all5,886 predictions, targets and per-row losses exactly, all100 seed7 batch-order hashes and1,800optimizer steps: PASS. One initial SSH timeout (attempt123030) occurred before any run directory/session or training was created; confirmed absent before retry. No scientific training reruns. Local D ENOSPC blocks fetch/full local copies; remote project and GitHub hold durable evidence.

Commands: `python -m pytest research_log/T059M/test_core.py -q -p no:cacheprovider --import-mode=importlib`; `python -m research_log.T059M.run train --out "$AUTODL_RUN_DIR/artifacts/T059M"`; `python -m research_log.T059M.run evaluate --out "$AUTODL_RUN_DIR/artifacts/T059M"`; `python research_log/T059M/verify.py "$AUTODL_RUN_DIR/artifacts/T059M"`. OMP/MKL/OPENBLAS threads1; CPU required by frozen authorization despite user general GPU preference.

All new-source-image, feature, reference-gradient, legacy/detail-gradient reads, outer, target-domain, LOL-v2, official-test and inference-reference counters zero. Stop for research-lead review; no architecture/loss search or rollout.
