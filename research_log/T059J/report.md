# T059-J — DONE / REFERENCE_ORACLE_ONLY

`even oracle choice among the frozen five donors cannot recover held scalar geometry; local 28-D scalar support is inadequate on inner-held images`.

|Query|Oracle-floor Huber|Gate margin|Verdict|
|---|---:|---:|---|
|Train-LOO4357rows|0.05390169471502304|+0.022606803104281425|PASS|
|Inner-held1529rows|0.21286551654338837|-0.1363570187240839|FAIL|

The fixed gate is0.07650849781930447. Even source-target oracle selection inside this exact five-donor neighborhood does not recover held scalar geometry. This is a local-support diagnosis for the frozen representation/neighborhood only, not a claim that every possible neighborhood or estimator fails. No selector, larger donor set, new search, new features, training or rollout was attempted. Oracle donor choice must never enter deployment/test-time logic.

Authorization857d3542a3e8872049d3cce23d92c5e9d4ff0869; source71cfffd77d6af3206bdafa46aee41334eaeccec2. Sole CPU audit20260918-075858-ttie-t059j-oracle, 2026-09-17 23:59:02–23:59:07UTC, exit0. Server2tests1.41sPASS; separate verifierPASS, independently recomputing all5886 oracle ranks/losses,29430 donor values, histograms, range/IQR/coverage and classification. Frozen order resolves all ties; no subgroup contributes to classification.

G1NN Hubers exactly replay0.06445551663637161/0.23172274231910706; Iconsensus exactly replay0.08473703265190125/0.2230137139558792. I maps/predictions/result and original source bindings match accepted hashes; every donor global ID/image ID/order matches the frozen map, all queries have5distinct images, trainownimageexcluded. No neighbors recomputed. Accepted G normalization/features remain unchanged and their artifact hashes are bound.

Oracle-best rank1fractions train0.3158136332338765 / held0.2707652060170046. Rank1–5histograms train[1376,941,807,602,631], held[414,362,314,228,211]. Target-in-donor-range coverage train0.7236630709203581 / held0.7109221713538261. All descriptive scalar range/IQR distributions are retained in result.json. In-range coverage does not imply a donor itself is accurate; interpolation is not authorized here.

Only training scalar storage is read to construct/persist the donor-value tables for all train/held queries. Then a separate read-only evaluation process opens held target storage and computes the explicitly source-only oracle floor. The same boundary-preserving replay ordering accepted in I is used: initial recorded baseline values from hash-bound evidence, rowwise baseline replay only after donor-table freeze. Whole mixed Gscalar-file hashing is deferred until held evaluation; selected training bytes are checked before/after. No detail-gradient tensors deserialized. The oracle comparison uses source targets solely in evaluation and is labelled REFERENCE_ORACLE_ONLY throughout.

All before/after I/G input hashes and six current source bindings unchanged; original I source bindings verified again after evaluation. Training/optimizer/model/newneighbor/newimage/newfeature/newreference/detailtensor/outer/target/LOL-v2/test/inference-reference counters0. C2outer16images/1460rows remain unopened. Initial local authorization write failedENOSPC before science; complete source/auth persistedremote and committed before execution. No scientific failures/reruns. Localworktree incomplete; serverhome/F and GitHub authoritative.

Donor values persisted UTC 2026-09-17T23:59:04.022196+00:00 SHA256 9700b4732c0e4a8e222bbe6c72b98936a2f349b7091c8ec3d389cffd9abc4b78; held scalar opening UTC 2026-09-17T23:59:05.631724+00:00.

Commands: python -m pytest research_log/T059J/test_core.py -q -p no:cacheprovider --import-mode=importlib; python -m research_log.T059J.run freeze/evaluate --out <run>/artifacts/T059J (separate invocations); python research_log/T059J/verify.py <run>/artifacts/T059J. Stop for research-lead review.
