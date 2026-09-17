# T059-C — BLOCKED before training: manifest/split mismatch

The required split fails its explicit preregistered checks. No training was attempted and no source supervision tensor was loaded. This is not a negative generalization result.

| Manifest-only check | Required | Observed |
|---|---:|---:|
| Manifest bank entries | 80 | 400 |
| Unique source images | 80 | 80 |
| Bank entries per image | implied one | exactly five for each image |
| Held-out / training banks under `bank_index % 5 == 0` | 16 / 64 | 80 / 320 |
| Held-out / training distinct images | 16 / 64 | 80 / 80 |
| Image-ID overlap | 0 | 80 |
| Held-out / training rows | complete disjoint image groups | 218 / 7,128 |
| Canonical row partition | all 7,346 exactly once | PASS |

For example, entries0–4 all belong to image36660; entry0 goes to holdout and entries1–4 to training. Entries5–9 all belong to image36678 and have the same leakage pattern. All80 image IDs overlap. The observed400-bank structure is the accepted frozen T014 structure, not a new data version: manifest SHA256 `92fd60d01ca0780880d724464c4d0a76ba1721ab5b8a2d054d4035a141673125` matches its accepted binding.

Authorization `8fe42e83182a57e490406331dcacc981343542e1` explicitly says any split/binding mismatch stops before training. Accordingly, no regrouping or substitute split was executed. Recommend that the research lead explicitly authorize a stable unique-image grouping in first-occurrence manifest order, then specify whether `image_group_index % 5 == 0` assigns all five constituent banks to holdout. That would be a revised task, not an interpretation silently applied here.

Evidence: `audit_split.py` is a manifest-only standard-library script. Command: `python3 /home/wenchang/asdasdsad/wjq/TTIE/shared/t059c/audit_split.py`. It ran successfully, verified the manifest hash, and generated explicit bank indices, image IDs, row counts, canonical row assignments and all five split-check booleans. No torch/model/target/reference files were read. Source SHA256 `e19da651190a155e5aa9af827c6b4098e615c59d54adec758276396462f64ecc`; exact audit bytes are gzip-compressed and hash-verified locally. Raw manifest/source/audit archive is verified in home/F; details are in T059C_archives.json.

Counters: training_runs=0; optimizer_steps=0; supervision_tensor_loads=0; heldout_supervision_reads_before_checkpoint=0; new_source_image_opens=0; reference_gradient_recomputations=0; new_feature_forwards=0; target_domain_access=0; lolv2_access=0; official_test_access=0; inference_reference_leakage=0. No final checkpoint, epoch history, fitted metrics or generalization verdict exists for T059-C. Original manifest remains unchanged.

PR99 automated review suggests recomputing verifier booleans instead of copying recorded flags. The five accepted BF numerical values and margins are consistent with all reported booleans. This is a hypothetical verifier-hardening suggestion rather than an observed BF verdict failure; no historical scientific/evidence code was changed during this manifest audit. Research lead already accepted BF.

Stop for research-lead split correction. PROJECT_STATE.md remains untouched. No self-merge or real-domain rollout.
