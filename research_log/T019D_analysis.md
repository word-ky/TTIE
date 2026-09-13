# T019-D fresh qualification positive

All five predeclared clauses pass on exactly40 new source images /120 episodes. The frozen T019-C selector was applied once, without training, tuning, replacements or a second cohort. This is the requested one-shot fresh qualification; no T020/corrective experiment has begun.

## Results

| Group | n | H0 | H1 | H* | H1/H0 | H1/H* | Beneficial/equal/harmful |
|---|---:|---:|---:|---:|---:|---:|---|
| spatial_pool | 120 | 0.032368708619227013 | 0.030830033973325045 | 0.030005912009316186 | 0.95246413244339334 | 1.0274653196261119 | 43/73/4 |
| left_right | 40 | 0.029433781816624104 | 0.029554176970850676 | 0.028184275573585181 | 1.0040903732648645 | 1.0486051661568834 | 13/23/4 |
| quadrants | 40 | 0.030063052033074199 | 0.030063052033074199 | 0.030063052033074199 | 1 | 1 | 0/40/0 |
| offset_left_right_40 | 40 | 0.037609292007982728 | 0.032872872916050254 | 0.031770408421289177 | 0.87406252978845889 | 1.0347009859030432 | 30/10/0 |

Literal acceptance vector: **[true,true,true,true,true]**. Pooled H1/H0<=.97; pooled H1/H*<=1.03; offset H1/H0<=.95; left/right and quadrants H1/H0<=1.01. No rounded-threshold relaxation.

Pooled MSE improves4.7536% over H0 and lies2.7465% above the nine-hard oracle. Quadrants make0 moves and have0 harmful cases; all40 remain canonical center. Left/right mean MSE is0.4090% above H0, within the predeclared1% tolerance; all4 remaining harmful cases are left/right. This is not an improvement on every family. No matched T018-D comparator was replayed, so this run establishes the prescribed qualification on a new cohort, not a matched fresh causal effect of changing labels. T019-B remains the controlled development intervention.

| Group | No move | x-only | y-only | Both |
|---|---:|---:|---:|---:|
| spatial_pool | 73 | 19 | 17 | 11 |
| left_right | 23 | 2 | 13 | 2 |
| quadrants | 40 | 0 | 0 | 0 |
| offset_left_right_40 | 10 | 17 | 4 | 9 |

Total47 moving episodes (19 x-only,17 y-only,11 both),73 no-move. Quadrants moving/harmful case table is empty. Remaining harmful cases:

| Row | Image ID | bx | by | H0 | H1 | H* |
|---:|---:|---:|---:|---:|---:|---:|
| 51 | 68286 | 0.4 | 0.4 | 0.040237236768007278 | 0.051612410694360733 | 0.035233203321695328 |
| 72 | 69213 | 0.4 | 0.5 | 0.031997155398130417 | 0.04472656175494194 | 0.029446357861161232 |
| 75 | 69224 | 0.4 | 0.5 | 0.009850040078163147 | 0.015887884423136711 | 0.0096868006512522697 |
| 78 | 69356 | 0.4 | 0.6 | 0.019846225157380104 | 0.021300656720995903 | 0.016688352450728416 |

## Cohort and immutable pipeline

Exclusions: **745 IDs** = existing T018-E exclusion705 + all40 accepted T018-E manifest IDs. Metadata audit covers33 tracked manifest/exclusion artifacts; no additional inspected ID was found. Every input records exact commit/path/SHA and extracted IDs. The T018-E manifest origin is explicitly accepted merge05f9f5a70b4c4d441c1f0d701ae8d64702e7052b. No T018-E features, logits, reference metrics or outcomes were used to develop or adjust the method.

The manifest selects the first40 numerically ascending eligible COCO val2017 IDs after that union; original min-side>=320 and the original source-pool/eligibility function are unchanged. All40 are disjoint from745 exclusions. Each yields left_right,quadrants,offset_left_right_40 in fixed order. No second draw, replacement or cohort shopping occurred. Manifest SHA **ca3d7aca2cf7e6224e11e234dff63cf3cac44e47f749aaf67d70f9e76ac5b391**. The manifest and independent verification list all40 IDs. These IDs are now used and unavailable for future corrective fresh tuning.

Scientific source **2e4f8bfa81ba8edd9cd1804f724a4747f1318847**. All68 baseline scientific files remain byte-identical to the accepted T018-E/T014 lock. Canonical hard Region2 trajectory from identity,40 label-free projected updates when active, frozen T006/T007 gate and T014 Sobolev energy, five hard-cross coordinates and28-D feature computation are unchanged. Only the selector lock changes to merged T019-C1714188c39dfff38986689cfbfb1671512d4c37f. Its receipt remains **0367456d7b4f235f987339baf07e343f86f870d1e109adbe461297c82b641c77**, and both checkpoints are hash-checked without rewriting.

GPU feature run **20260913-174415-ttie-t019d-fresh-features**; separate evaluation run **20260913-175047-ttie-t019d-postfreeze-eval**. Release20260913-174150-ttie-t019d-frozen. PhysicalGPU1 NVIDIA RTX A6000 / Torch2.4.0+cu121 runs the existing image/CLIP/TTT and renderer evaluation. The primary heads run on original Windows Python3.12.7/Torch2.13CPU, in one120-row batch, to preserve exact frozen inference. No selector training or normalization refit.

## Information order and bindings

| Event | UTC |
|---|---|
| Exclusion union frozen | 2026-09-13T09:32:55.470483+00:00 |
| Manifest frozen | 2026-09-13T09:44:21.906105+00:00 |
| Offline degraded-only synthesis complete | 2026-09-13T09:44:24.917853+00:00 |
| GPU feature start | 2026-09-13T09:44:31.037796+00:00 |
| Features frozen | 2026-09-13T09:48:36.701672+00:00 |
| Primary decisions frozen | 2026-09-13T09:49:38.112406+00:00 |
| Independent reference-free replay | 2026-09-13T09:49:58.123379+00:00 |
| Reference evaluation opened | 2026-09-13T09:50:54.752707+00:00 |
| Independent metrics verified | 2026-09-13T09:52:19.878332+00:00 |

Features SHA **2cbe26e2b9231eaf2fba8d76dd7e699e1531e4f161ce13c878591684a70a9851**; primary decisions SHA **438350e17e9a3b5d0b3e64490f68c3f543d2ecf70f8e93396748f34c4ca488f8**. All120 logits/classes/bx/by exactly reproduced independently before reference opening.

The pre-feature config contemporaneously binds all six required preparation artifacts; it is then pinned by feature and decision freezes:

| Binding | SHA256 |
|---|---|
| exclusions_sha256 | 3b107d77a0d1cee1d2a9ee23f434c8a5d55022b1a51d2d9b4f1d6a810ededaa6 |
| manifest_sha256 | ca3d7aca2cf7e6224e11e234dff63cf3cac44e47f749aaf67d70f9e76ac5b391 |
| manifest_frozen_sha256 | db4955d9f81b87c5da9909cf1bcfee2a87f5a1b908f34f4310589924fcbf8aff |
| mapping_sha256 | afdb45de0a39ab5a31216b61ed8bd335f07ce0e878e68cc6aea11595f9df713d |
| input_index_sha256 | 00b3d45e729973a394b11493b7e1cfb8f84b887e23f2903624e1403c60f0687d |
| prepared_sha256 | abf02a420ad0904c38ac7d5ec496969972556c157089bdd512c63bb692fcce95 |

Preparatory manifest/mapping files are byte-hashed for integrity before the deployment read barrier; they are never deserialized into family/semantic features. Offline corruption synthesis necessarily reads source pixels only in its separate process after manifest freeze; it produces opaque degraded-only tensors. The model path consumes those tensors/frozen assets and five label-free features, never clean/reference pixels, targets, masks/gain maps, family/condition IDs or image-ID semantics. The inference API exposes only the five feature vectors. Raw metadata hashing does not provide a model input.

The evaluator requires an independent replay receipt matching all120 frozen decisions before opening clean/reference pixels. The independent metric verifier checks all33 original exclusion commit:path inputs;745-ID union; first40 ordering/eligibility/disjointness; exclusion→manifest→prepared→feature→decision→replay→reference timestamps; all78 scientific/source files; exact manifest-to-mapping order; mapping/index/preparation/config hashes; every feature/decision input and case-file binding; all120 nine-candidate table lookups and family/pooled arithmetic; literal5 clauses; outcomes and movements. This recomputes stored-table arithmetic, not a second pixel run.

## Validation and limitations

Baseline9tests pass22.08s. Final affected16tests pass17.28s, including six missing-preparation-binding failures, mapping edits/refreshed preparation rejection, feature equivalence, reference-read barrier, missing independent-replay rejection and literal gate boundaries. A development-only smoke through the new CPU wrapper and independent verifier exactly reproduces the original frozen C120-row replay; it uses no fresh/E data. Both formal remote runs exit0; original CPU inference and independent replay/metric processes exit0. No performance failure or scientific deviation.

Two pre-launch packaging issues were observed and repaired before selecting any new images: the shell script working-tree/index newline state, restored to exact committed bytes; and missing Git tree objects needed by the remote source-status check, restored from the actual source commit. All78 source/assets/selector checks then passed unchanged. The existing NVML warning did not affect CUDA; drivers and unrelated jobs were left alone.

No T018-E fresh data influenced selection or tuning; only its image IDs/manifest metadata enter the required exclusion union. No fresh label or clean target entered adaptation or selection. This result is confined to the specified40-image/three-condition protocol and does not establish universal safety or downstream benchmark performance. Stop here for research-lead review; do not alter PROJECT_STATE.md or start a corrective/T020 experiment.
