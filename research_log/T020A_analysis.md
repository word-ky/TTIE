# T020-A fresh safety negative

The sole 40-image / 120-episode fresh non-spatial safety qualification passes **3/4** literal clauses. The clean clause fails. No method change, tuning, selector refit, second cohort or corrective experiment was performed. All scientific processes completed successfully; independent replay/provenance/metric checks pass. Software correctness does not turn this scientific negative into a positive.

## Results

| Group | n | H0 MSE | H1 MSE | H1/H0 | Beneficial/equal/harmful | Literal <=1.01 |
|---|---:|---:|---:|---:|---|---|
| nonspatial_pool | 120 | 0.023662558179057668 | 0.023345857432286721 | 0.98661595486107501 | 16/98/6 | True |
| clean | 40 | 0.00082311523146927357 | 0.00086108171381056304 | 1.0461253551018836 | 0/39/1 | False |
| homogeneous_dark | 40 | 0.051679096417501569 | 0.051374320825561884 | 0.99410253636252688 | 4/34/2 | True |
| homogeneous_bright | 40 | 0.018485462888202166 | 0.017802169757487718 | 0.96303619039204369 | 12/25/3 | True |

The pooled mean improves about 1.3384%, but clean mean MSE increases **4.6125%**, exceeding its predeclared 1% tolerance. Clean has six x-only boundary moves: five leave MSE equal and one worsens it. The harmful clean episode is row15, image73533, selected `(bx,by)=(0.4,0.5)`: H0=0.0116857485845685, H1=0.013204407878220081, absolute increase0.0015186592936515808. All other39 clean episodes are equal; there is no clean beneficial episode. This single row accounts for the clean mean increase of0.00003796648234128952. The overall improvement and unchanged clean p95 do not override the failed clean-mean clause.

| Group | No move | x-only | y-only | Both |
|---|---:|---:|---:|---:|
| nonspatial_pool | 93 | 15 | 9 | 3 |
| clean | 34 | 6 | 0 | 0 |
| homogeneous_dark | 34 | 0 | 5 | 1 |
| homogeneous_bright | 25 | 9 | 4 | 2 |

| Clean absolute MSE | Mean | p95 |
|---|---:|---:|
| H0 | 0.00082311523146927357 | 0.0098961647134274203 |
| H1 | 0.00086108171381056304 | 0.0098961647134274203 |

p95 uses linear interpolation at `0.95*(n-1)` among40 sorted episode MSEs, fixed in source before the manifest. Ratios are null when an individual H0 is exactly zero; acceptance always compares absolute means directly, with no epsilon or threshold relaxation.

All harmful cases:

| Row | Image | Condition | bx | by | H0 | H1 |
|---:|---:|---|---:|---:|---:|---:|
| 15 | 73533 | clean | 0.4 | 0.5 | 0.011685748584568501 | 0.013204407878220081 |
| 17 | 73533 | homogeneous_bright | 0.4 | 0.5 | 0.01340020727366209 | 0.013612456619739532 |
| 76 | 76731 | homogeneous_dark | 0.5 | 0.4 | 0.031391862779855728 | 0.033417243510484695 |
| 95 | 78266 | homogeneous_bright | 0.4 | 0.5 | 0.023131994530558586 | 0.024861333891749382 |
| 98 | 78420 | homogeneous_bright | 0.4 | 0.5 | 0.023093277588486671 | 0.023182043805718422 |
| 112 | 78843 | homogeneous_dark | 0.4 | 0.4 | 0.037252865731716156 | 0.039656654000282288 |

## Frozen method and cohort

Scientific source `49888ea2a8f6cb30cf292ecffb4ac715066901d2`, based on main `cd59bf24` accepting merged D `f50a3b6a027f647efebf46183e934f8bba423882`. All68 baseline scientific files remain byte-identical. Preparation, selection and inference reuse merged D code unchanged apart from task namespace/path names. The same gate, Sobolev energy, canonical hard Region2 trajectory from identity, exactly40 projected label-free updates when active, checkpoint rule, five hard-cross coordinates and28-D features remain fixed. C selector receipt is unchanged `0367456d7b4f235f987339baf07e343f86f870d1e109adbe461297c82b641c77`; original two heads/normalizers are never rewritten or refit.

The conditions are the original T014 `clean`, `homogeneous_dark`, `homogeneous_bright`, in that order, from byte-locked `ttie/natural.py`: gains1,0.45,1.55, with the existing clipping and image preprocessing. Evaluation renders only canonical `(0.5,0.5)` and the frozen chosen hard boundary. No nine-way oracle search, extra feature candidate or downstream experiment is run.

The exclusion audit binds36 tracked original commit:path metadata artifacts, including empty-ID provenance manifests. The union is788 IDs: D's prior745 plus all40 D selected IDs plus the three D-prefix metadata-inspected but ineligible IDs67534,68409,68933. E and D cohort origins explicitly use their accepted merges. All historical inspected/used IDs enter the union; no E/D feature, logit or reference result was used to adjust this method. The new40 IDs are the first eligible numeric IDs after the union under the original min-side>=320 rule, selected exactly once. No replacement or second cohort occurred. These40 and any additionally inspected prefix IDs are now burned for future corrective fresh tuning; preserve the complete manifest.

Manifest SHA `da2fe6837d4822b1f22888e2144ded663e0c2d43faa78b1461bf8a19093cca24`; features SHA `38f66d723cfe88f65aea1f19bd17da4bd6b2cca4de36d58c24f80f4d81599537`; decisions SHA `83b3797dafed317f94884ff16a9269f4249457534e5ac3442b489f419cca0ac0`. All40 fresh IDs and provenance are listed in the manifest and independent verification receipt.

## Information order and bindings

| Event | UTC |
|---|---|
| Exclusions frozen | 2026-09-13T10:27:35.397752+00:00 |
| Manifest frozen | 2026-09-13T10:30:39.649707+00:00 |
| Offline synthesis/preparation complete | 2026-09-13T10:30:42.781902+00:00 |
| GPU features start | 2026-09-13T10:30:49.581194+00:00 |
| GPU features frozen | 2026-09-13T10:33:34.882747+00:00 |
| CPU decisions frozen | 2026-09-13T10:34:35.422985+00:00 |
| Independent reference-free replay | 2026-09-13T10:34:41.275800+00:00 |
| Reference evaluation opens | 2026-09-13T10:35:20.077685+00:00 |
| Independent metric verification | 2026-09-13T10:36:30.727962+00:00 |

The config created before the first feature row binds all six preparation artifacts, then feature/decision receipts pin the same config.

| Binding | SHA256 |
|---|---|
| exclusions_sha256 | 2084931d52db0d87f02aa79132874b018c7e1921593042e6028ac3aafd325763 |
| manifest_sha256 | da2fe6837d4822b1f22888e2144ded663e0c2d43faa78b1461bf8a19093cca24 |
| manifest_frozen_sha256 | 6bde1d9e8d35a5d6e482ad3e0267a1544cca61b6d3e42fc9e2134e88d67cf381 |
| mapping_sha256 | 7fbf1d2076da384fbeb441fe15817b8e753bb5baa12e61340a4784fc618a435d |
| input_index_sha256 | 5276acbbd29fb1128eb024fcd0bdd4a4ab8db4e8bb087fe16e84781d0363ce03 |
| prepared_sha256 | 46ca0933f11a0ae0a799e8655a5ddb6a323d017ecd6a91d34da8d2d9a3114358 |

Manifest/mapping bytes are hashed only for integrity, before the deployment read barrier, and are not decoded into model inputs. Offline corruption synthesis uses source pixels in its own process. The adaptation/selector path consumes degraded-only tensors and frozen assets/features, never clean targets, reference metrics, family labels, masks/gain maps, annotations or semantic image-ID inputs. For the clean condition the opaque input is naturally an uncorrupted image; the deployment path neither knows that label nor opens a second clean target.

All120 logits/classes/boundaries were independently reproduced exactly before reference opening. The independent verifier imports no TTIE code for neural replay and performs no normalization refit. Its post-freeze phase validates all36 original metadata origins, the788-ID exclusion union, exact first40 ordering/disjointness/eligibility, full freeze chronology, all78 source files, preparation/config/mapping/index/feature/decision/case hashes, all120 row correspondences, two-column MSE arithmetic, outcomes/movements and clean mean/p95. This checks stored metric arithmetic, not a second pixel experiment.

## Execution and validation

Feature run `20260913-183032-ttie-t020a-safety-features`; evaluation run `20260913-183512-ttie-t020a-postfreeze-safety`; release `20260913-182948-ttie-t020a-frozen`. Both remote jobs exit0, as do original CPU prediction and both independent verification phases. A6000 physicalGPU1 / Torch2.4+cu121 executes image/CLIP/TTT and evaluation; Windows Python3.12.7 / Torch2.13+CPU executes only the tiny frozen heads to preserve the original backend's exact replay.

Baseline16 tests passed17.25s; first changed increment18 tests passed18.56s; final affected24 tests passed36.18s. Tests cover literal original condition gains, exact feature reuse, all six preparation bindings, mapping swap/refreshed-preparation rejection, reference-read barriers, independent-replay prerequisite, four exact safety thresholds, zero-clean-MSE arithmetic and p95. All68 donor files and namespace-only prepare/select/predict equivalence were separately checked. Remote78-file source/assets/heads preflight passes. The known NVML warning did not prevent CUDA; one initial status SSH read exited255, followed by a successful bounded directCUDA/status check. No scientific run failed or was rerun.

No scientific design deviation, confidence/fallback rule, family-specific handling, deadband change, second seed, real low-light benchmark or detector evaluation. The only scientific failure is the predeclared clean safety clause. Code and source provenance were frozen before the unique manifest. Compact archive hashes were verified before local extraction; full degraded tensors remain remote for mirrored recovery.

Recommendation: retain T014 as the broad deployable baseline; T019-D remains heterogeneous-only fresh-positive, but T020-A does not authorize promotion of T019 to a broadly safe default. Stop for research-lead review. Do not recalibrate or run T020-B/corrective experiments in this cycle, and do not modify research-owned PROJECT_STATE.md.
