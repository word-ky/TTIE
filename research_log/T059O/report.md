# T059-O-R — DONE (unchanged T059-O scientific audit)

explicit bank-anchor context is insufficient for source-image scalar localization under fixed 1-NN.

Source `52d4640e6faf5693d0568e37ed173d5e2c9955bf`; recovery authorization `9b68e1ed30bf2f5dc663ffa2f529eddf18b2b255`; original authorization `93c4ab9f9681a9c5d445a4244d3a82ed5b6e4c44`.

| Metric | Fit-LOO | Selector |
|---|---:|---:|
| huber | 0.04802930727601051 | 0.1556149274110794 |
| margin | 0.028479190543293953 | -0.07910642959177494 |
| rows | 3604 | 753 |
| queries_with_ties | 125 | 21 |
| maximum_ties | 4 | 3 |
| candidate_count_min | 3507 | 3604 |
| candidate_count_max | 3553 | 3604 |

Fixed gate0.07650849781930447: fit source control passes, selector fails. No further representation, weighting, k, metric, or model attempted. Selector is the already-observed N mechanism cohort, not fresh confirmation.

Exactly N40fit/8selector images,3604/753rows,200/40banks; each bank has one state0 anchor. N fit-only normalization is SHA-bound and replayed exactly from fit raw features. z=concat(u-u0,u0),float32 construction; no re-standardization or half weighting. GPU1A6000 direct float64 squared Euclidean k1 reuses Gnearest; exact ties smallest canonical global row. Fit donor pool excludes own image; selector uses all3604fit rows. All donor global and image IDs, counts and ties in maps.pt; per-image/per-bank summaries in result.json.

Vectors/anchors persisted 2026-09-18T07:38:31.199487+00:00; complete maps persisted 2026-09-18T07:38:31.747106+00:00; fit scalar opened 2026-09-18T07:38:33.470025+00:00; donor predictions persisted 2026-09-18T07:38:33.500701+00:00; selector scalar opened 2026-09-18T07:38:35.044219+00:00. Vectors SHA 3af0bf39240777e97e69a97d1df001f02648b0fa4ec01a232f4206c27490c74e; maps SHA b238dec39c1e72b8137d5a67112ae054fe6c5f3b087f7fec07279f447275766c; predictions SHA 362f99a59e3ca84524635f6eb9c2fa103ae5504ef593116209c6ddbc6c5396eb. Hashes of u/u0/z/anchors/normalization and all accepted E/N sources retained. Before/after inputs and outputs unchanged.

Four focused tests passed1.39s. Separate preflight verified source bindings,40/8partition,fit-only normalization,unique anchors and verifier syntax/dependencies before audit; no scalar reads. Sole run20260918-153825-ttie-t059or-context,07:38:29–07:38:41UTC exit0. Independent CPU NumPy replay verifies all4357vectors,fullcandidate distances,ties,neighbors,donor IDs,predictions/losses and all per-image/per-bank summaries PASS; maximum distance differences7.105427357601002e-15 /3.552713678800501e-15, exact neighbor/tie choices.

Original combined verifier/auth/test command was policy-rejected before any execution, reported in c4365fbacbcd80bf674fe88114b1471ea10d5bf3. New O-R instruction explicitly requested atomic normal steps; verifier write, authorization save, binding, tests, preflight, source commit and audit were performed separately without another policy rejection. Two SSH connection timeouts affected preflight/log read only; reconnected, no scientific rerun. Nonblocking NVML warning. LocalD ENOSPC blocks fetch/full copies; remote project/GitHub preserve records.

Commands: `python -m pytest research_log/T059O/test_core.py -q -p no:cacheprovider --import-mode=importlib`; `python -m research_log.T059O.run maps --out "$AUTODL_RUN_DIR/artifacts/T059O"`; same module stages `predict` then `evaluate` in separate processes; `python research_log/T059O/verify.py "$AUTODL_RUN_DIR/artifacts/T059O"`. CUDA_VISIBLE_DEVICES=1,OMP/MKL/OPENBLAS threads1.

All training/optimizer/model,selector-before-map/prediction-freeze,inner-held,outer,newsource/newfeature/refgradient,legacy/detail tensors,target-domain,LOL-v2,official-test and inference-reference counters0. Stop for research-lead review; no head training or later cohort.
