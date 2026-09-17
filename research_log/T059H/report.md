# T059-H — DONE

`nearest-distance shift alone does not explain T059-G scalar failure; conditional scalar mismatch remains`.

Distance-reweighted train Huber **0.06854875106125766** remains below the fixed **0.07650849781930447** gate, while actual inner-held Huber is **0.23172274231910706**. Original train Huber **0.06445551663637161** and held Huber replay exactly. This fixed ten-bin attribution does not explain the held-out scalar error by nearest-distance proportions alone. It does not establish a causal feature defect or authorize adding image-conditioned features, changing the loss, or deployment.

Authorization44476e52a27cd906d3132da1dba38df390a3f67d; tested source799b1a1fc025ebb08bd74d13f94ce01eca5ad5bb. Sole CPU run20260918-063357-ttie-t059h-deciles, 2026-09-17 22:34:03–22:34:06UTC, exit0. Server2focused tests1.29sPASS; separate verifierPASS and identical reweighted value/classification. Local ENOSPC prevented source creation, so local pytest found no test file and no tests ran. Complete source was persistedremote and published before scientific execution. No scientific rerun/failure.

Only the three accepted G artifacts (maps.pt, evaluation_rows.pt, result.json) were read. Exact SHA256s, map/row/scalar tensor hashes and before/after checks are in result.json. Persisted global IDs, neighbor indices, cross-image exclusion, scalar prediction identity and all4357train/1529held rows are replayed. No neighbor search or fresh feature/gradient/target-source access occurred. Existing G artifacts contain other persisted tensors, but only maps and scalar pairs are used; nothing is regenerated. Excluded C2 outer supervision remains unopened. All requested counters0. Five source bindings unchanged.

Quantiles are deterministic empirical type7 linear interpolation at p=.1,...,.9 using position=(n-1)*p and float64 train-LOO squared distances only. Boundaries are persisted/fsynced before scalar artifact access and any bin-loss calculation. Bins use [-infinity,q10),[q10,q20),...,[q90,+infinity], so internal-boundary ties go higher and every row is retained. No bin or gate tuning, trimming, alternate weighting, subgroup rescue or second classification. Float32 G per-row Huber contributions are accumulated in float64 for per-bin means/reweighting. Independent verifier computes sorted-order quantiles, NumPy bin assignments and Huber contributions without importing the audit implementation.

Held fraction strictly above train p90: **0.1327665140614781**; above train maximum: **0**. Full train/held distance summaries and exact boundaries are in result.json. Counts/means/weights follow (bin IDs0–9):

|Bin|Train rows|Held rows|Train mean Huber|Held mean Huber|Held weight|
|---|---:|---:|---:|---:|---:|
|0|435|87|0.005727456831576451|0.0029869150315885783|0.05689993459777633|
|1|437|151|0.002829846125464525|0.1904838538802775|0.0987573577501635|
|2|434|94|0.0049131432860789435|0.433311796153838|0.061478090255068674|
|3|436|81|0.010981336907259488|0.4404596786580638|0.052975801177240024|
|4|436|166|0.007550457079109152|0.45807694712469715|0.10856769130150425|
|5|434|297|0.014697359149952573|0.05827595132103763|0.19424460431654678|
|6|438|185|0.04088449867286449|0.13558377173469385|0.1209941137998692|
|7|435|131|0.2852331031695199|0.9126532639346503|0.08567691301504252|
|8|436|134|0.04638641586071188|0.014842563311072083|0.08763897972531066|
|9|436|203|0.2254713353069003|0.04381483243524367|0.1327665140614781|

Boundary persistence UTC: 2026-09-17T22:34:04.498754+00:00; scalar pair access UTC: 2026-09-17T22:34:04.499397+00:00.

Commands: python -m pytest research_log/T059H/test_core.py -q -p no:cacheprovider --import-mode=importlib; python -m research_log.T059H.run --out <run>/artifacts/T059H; python research_log/T059H/verify.py <run>/artifacts/T059H. Stop for research-lead review. Local worktree is incomplete dueDspace; GitHub and remotehome/F are authoritative.
