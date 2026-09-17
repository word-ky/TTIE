# T059-C2 — image-held-out dual-tangent generalization not supported under fixed split

The one fixed fit completed normally, but held-out detail median cosine and value Huber fail their preregistered gates. The other three gates pass. No second split, seed, tuning, full-data refit or real-domain rollout followed.

| Held-out statistic | Training diagnostic | Held-out actual | Fixed threshold | Margin | Pass |
|---|---:|---:|---:|---:|---|
| detail_positive | 0.98690783977508545 | 0.8450312614440918 | >= 0.75 | 0.095031261444091797 | True |
| detail_cosine | 0.64418965578079224 | 0.4797055721282959 | >= 0.5 | -0.020294427871704102 | False |
| legacy_positive | 0.99397385120391846 | 0.95069444179534912 | >= 0.94999999999999996 | 0.0006944417953491655 | True |
| legacy_cosine | 0.96816337108612061 | 0.93803977966308594 | >= 0.90000000000000002 | 0.038039779663085915 | True |
| value_huber | 0.071759872138500214 | 0.19325308501720428 | <= 0.076508497819304466 | -0.11674458719789982 | False |

Training diagnostics do not determine acceptance. Held-out legacy eligible/ineligible1,440/20; detail1,439/21. Training legacy5,808/78; detail5,805/81. All1,460held-out rows retained, including ineligible rows for value evaluation.

## Frozen split and information boundary

Authorization `7becddc0693e60ede79455ef547847bff4512e4c`. ManifestSHA `92fd60d01ca0780880d724464c4d0a76ba1721ab5b8a2d054d4035a141673125`:400entries,80unique images,exactly5entries each. Image groups use first occurrence in manifest order; groups0,5,...75 held out. Allbanks of eachimage stay together, without assuming contiguity. Training64images/320banks/5,886rows; heldout16images/80banks/1,460rows; zero image overlap and exact7,346-row partition. The ordered IDs, group indices, bank indices and canonical rows are in split.json.gz.

Only training banks targets.json/source_derivatives.pt were deserialized by training. AF reference chunks contain mixed-side rows; a metadata-only ZIP reader seeks directly to authorized float32 row byte ranges and materializes only5,886training reference rows. It never torch.loads a mixed reference chunk. SHA binding reads opaque file bytes; it does not deserialize held-out targets/reference tensors. Actual storage byte ranges and loaded indices are recorded. The selected-row reader test checks reads, and the grouping test uses noncontiguous banks. No new feature/Jacobian/reference generation.

Checkpoint saved, fsynced and hashed at `2026-09-17T16:30:51.554848+00:00`. Separate read-only evaluation process begins held-out loading at `2026-09-17T16:31:34.564979+00:00`. Independent verifier reopened all evidence/checkpoint, checked training/held-out accessed row sets and ordering, rehashed immutable assets, and independently recomputed all five booleans/classification from numeric values. heldout_supervision_reads_before_checkpoint=0.

## Fixed training and execution

Exact source `78fabc50b9bc03beb73888d056de5b91383cd22c`,223source/Gitblob bindings; unchanged T059B/fit.py and original train_head. Fresh seed7 CPU EnergyHead; normalization computed solely from5,886training rows. AdamW,batch256/order,100epochs,finalepoch and1:1:1value/legacy/detail loss unchanged. No BF warm-start. Exactly1training run/2,300optimizer steps. Full100epoch loss history and initial/final head hashes retained.

Sole run `20260918-002953-ttie-t059c2-holdout`, release `20260918-002848-ttie-t059c2-holdout`, exit0. Local5tests11.20s; server5tests5.54s. Train-stage runtime63.95960594096687s; separate evaluation40.45971866406035s; final independent verifier completed16:32:37.066776UTC. No execution failure or scientific rerun.

Command sequence: fixed one-thread CPU environment; `python -m pytest research_log/T059C2/test_core.py research_log/T059B/test_fit.py -q -p no:cacheprovider --import-mode=importlib`, then `python research_log/T059C2/run.py train --out <artifacts/T059C2>`, then a separate `run.py evaluate`, then `verify.py --out <artifacts/T059C2>`, connected with shell &&.

Final checkpoint SHA256 `df874a53aadd359c1363e5866179a251dbf5ca86ceb72988d53485873202d93c`. Raw archive 615212bytes SHA256 `ae1c57af910ce492379fc0e93c56dd2f650b244ef397c670668681af03a4cd0d`, verified home/F. Exact large JSON receipts are gzip-compressed; decompressed hashes match the raw archive/complete marker and were verified locally. Checkpoint resides in the raw archive, not a new model release.

All bound asset/input tensor hashes remain equal before/after. Counters: training_runs=1; optimizer_steps=2300; heldout_supervision_reads_before_checkpoint=0; new_source_image_opens=0; reference_gradient_recomputations=0; new_feature_forwards=0; target_domain_access=0; lolv2_access=0; official_test_access=0; inference_reference_leakage=0.

The fixed matched head fits its training images but does not meet the full held-out readiness criterion. The detail cosine misses by0.0202944 and value Huber exceeds the ceiling by0.1167446. This does not establish impossibility under every recipe, and it does not support deployment. Stop for research-lead analysis; PROJECT_STATE untouched, no self-merge.
