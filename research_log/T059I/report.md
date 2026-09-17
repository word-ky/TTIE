# T059-I — DONE

`fixed five-image consensus does not preserve inner-train scalar consistency; estimator smoothing is not a supported rescue`.

The first applicable classification is negative: train-LOO consensus Huber **0.08473703265190125** exceeds the unchanged **0.07650849781930447** gate (margin **-0.008228534832596779**). Held consensus Huber is **0.2230137139558792** (margin **-0.14650521613657475**), retained as a descriptive measurement, not a second classification. Original G 1-NN Hubers **0.06445551663637161 / 0.23172274231910706** replay exactly. No alternate k, weights, median, trimming, representation change, scalar-loss redesign or rollout was attempted. This fixed smoother failing does not prove every estimator would fail or establish a causal representation defect.

Authorizatione381a11cd04c3f8aa1e793fc66dd9ba16281d2be; tested source9440497880910118ad9fda53045971146959dbb9. Sole run20260918-073334-ttie-t059i-five, 2026-09-17 23:33:38–23:33:58UTC, exit0. PhysicalGPU1 NVIDIA RTX A6000 computed direct squared-distance five-image maps. Server3focused tests1.27sPASS; independent CPU verifierPASS, replaying all5886 maps /29430 donor selections and tie counts, exact predictions, chronological ordering and the same final classification. Maximum CPU/GPU distance differences1.0658141036401503e-14(train)/7.105427357601002e-15(held), all selections exact.

All4357train and1529held queries retain exactly5distinct training-image donors. Train queries exclude their own image completely; held queries use48eligible training images. First select each image's nearest row, breaking ties by smallest canonical row; then order candidate images by(distance,neighbor_global_row_index,training_image_id). First donors exactly replay original G1NN IDs/distances. Selected donors with within-image row ties507train/173held; queries with ties at fifth retained distance0/0. All maps, distances, donor IDs, tie counts, predictions and hashes are persisted.

The staged information boundary is explicit: map process reads only Gfeature-map and SHA-bound recorded result; prediction process opens only the train/target numeric byte range of G evaluation_rows.pt, averages5training scalars in float32, and persists/fsyncs/hashes all predictions; separate evaluation process then opens held targets. A selected-storage pickle metadata reader never deserializes detail-gradient tensors or incidental held storage. Whole mixed scalar artifact SHA is deliberately deferred until allowed held evaluation, then compared to the previously accepted G hash and rechecked after. Training bytes match before/after. Gnormalization provenance and exact standardized features are bound by accepted Gartifact SHA; no checkpoint/model forward or renormalization occurs.

The authorization asks for original baseline replay first but also forbids opening held targets before maps/predictions. To preserve the explicit information boundary, the pre-map replay uses SHA-bound recorded Gaggregate values; exact rowwise baseline recomputation occurs in evaluation before scientific classification. This execution ordering was documented in source progress.md before the run. Full mixed-file hashing was likewise deferred to avoid reading held storage early. No scientific gate was relaxed.

Only accepted G maps/result/scalar stores were used; C2outer16images/1460rows remained unopened. All training/optimizer/model/image/feature/reference/detail-tensor/outer/target/LOL-v2/test/inference-reference counters0. Before/after artifact hashes and five source bindings match. Initial local source write failed ENOSPC before science; complete source was persistedremote and committed before the sole run. KnownNVMLwarning was nonblocking; actualGPU is recorded. No scientific failures/reruns. Local worktree incomplete; GitHub and remotehome/F are authoritative.

Chronology and hashes:

- Maps persisted: 2026-09-17T23:33:48.769659+00:00; SHA256 289bee4e567a8daaa8180b2a7b58acbc4044ad26dbbe215aa92a818bf727ac81
- Training scalar opened: 2026-09-17T23:33:50.411355+00:00
- Predictions persisted: 2026-09-17T23:33:50.413134+00:00; SHA256 1a09e1b591d890223e0041f0d60cbddd112a716f25c888ac5c3862269dd534f6
- Held scalar access permitted/opened: 2026-09-17T23:33:51.945024+00:00

Commands: python -m pytest research_log/T059I/test_core.py -q -p no:cacheprovider --import-mode=importlib; python -m research_log.T059I.run maps/predict/evaluate --out <run>/artifacts/T059I (three separate invocations); python research_log/T059I/verify.py <run>/artifacts/T059I. Stop for research-lead review.
