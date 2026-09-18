# T059-N — DONE

fixed source-only early stopping does not establish image-held scalar transfer under the current 28-D EnergyHead.

Source `aced779b45ac857eddad6bc0ee4a2bee31c69a7f`; authorization `9ac9f1ffc3e0ec99f5f13ceaa78c9756a4bd0e87`.

All100epochs eligible (fit Huber <=0.07650849781930447). Selected epoch **13** by minimum selector Huber, exact ties earliest. Fit Huber **0.04073658958077431**, margin+0.03577190823853016; selector Huber **0.14937368035316467**, margin-0.0728651825338602. The selector fails. Epoch100 fit/selector: 0.0038375542499125004 / 0.17534901201725006. Full100epoch curves and eligible set in result.json; no alternative selection or retraining.

Parent48images sorted by accepted E canonical group_indices. Selector sorted ranks [5,11,17,23,29,35,41,47]; fit40images/200banks/3604rows, selector8images/40banks/753rows; disjoint image IDs and complete4357row parent coverage. Exact image/group/row/bank lists in split.json. Only fit files are opened during training; selector targets first opened by separate evaluator after freeze. No E inner-held or C2 outer feature/scalar files opened. Metadata-only parent split inspection does not load excluded supervision.

Unchanged M 28→64→64→1 SiLU EnergyHead, seed7, CPU float32, AdamW lr0.001, weight_decay0.0001, betas(0.9,0.999), eps1e-8, batch256, bank-relative Huber,100epochs/1500steps. Only fit rows determine x/y mean/population std (zero std→1). M loop changed only to add epoch checkpoint and exact full-fit Huber callback, no RNG change. fit.diff records the change. Checkpoint snapshots are evaluated without gradients.

All100head checkpoints, final optimizer/generator/history, fit normalization, initial/final hashes, fit row IDs and per-epoch fit predictions/metrics fsynced and SHA-bound at 2026-09-18T05:14:05.715891+00:00; selector scalar opening 2026-09-18T05:14:07.566493+00:00. Every checkpoint SHA is in checkpoints_persisted.json. Selected checkpoint SHA 1810e1a8e04e212f751fee9429eabe10bd918ff5f6add8b9bf5225d964bda32b. Immutable before/after source and inputs verified.

Tests2 passed in0.01s; accepted sources/fixed partition replay PASS,129 own source bindings. Sole CPU run20260918-131356-ttie-t059n-selector,05:14:00–05:14:10UTC exit0. Independent replay verifies split, all100checkpoint hashes, all435700 row predictions/targets/losses, fit-only normalization, all100batch orders,1500optimizer steps and selection rule: PASS. No scientific failures, startup failures or reruns. LocalD ENOSPC blocks git fetch/full local copies; remote project andGitHub preserve evidence.

Commands: `python -m pytest research_log/T059N/test_core.py -q -p no:cacheprovider --import-mode=importlib`; `python -m research_log.T059N.run train --out "$AUTODL_RUN_DIR/artifacts/T059N"`; `python -m research_log.T059N.run evaluate --out "$AUTODL_RUN_DIR/artifacts/T059N"`; `python research_log/T059N/verify.py "$AUTODL_RUN_DIR/artifacts/T059N"`. OMP/MKL/OPENBLAS threads1. CPU explicitly frozen by task.

Counters training_runs1, optimizer_steps1500; selector-before-freeze, E inner-held, C2 outer, new-source-image/feature/reference-gradient,legacy/detail-gradient reads,target-domain,LOL-v2,official-test and inference-reference all0. Stop for research-lead review; no confirmation cohort opened or deployment claim.
