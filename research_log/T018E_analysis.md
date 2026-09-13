# T018-E — one-shot fresh qualification negative

## Review amendment — historical preparation binding is incomplete

PR30 review3998852738 identified that the original frozen config pins the manifest and input index but omits prepared.json. The old mapping check accepts a modified mapping if its digest is refreshed in that mutable preparation record. The original pass receipt therefore does **not** establish the complete pre-inference mapping chain claimed below. The recorded MSE values and4/5 negative outcome are preserved; they must be read with this provenance limitation, not as a fully verified qualification protocol.

The repaired pipeline pins prepared.json in the pre-inference config and validates that digest, the mapping digest, and the input index during evaluation and independent verification. The strict verifier intentionally rejects the original run for its missing binding. No digest has been inserted into historical configs, freezes or receipts, and no scientific phase has been rerun.

A separate, explicitly retrospective audit confirms that all120 saved mappings exactly equal the Cartesian row order of the original hash-frozen40-image manifest and its three recorded conditions, and all120 input bindings match frozen decisions. This supports the consistency of the saved assignments; it cannot manufacture the missing contemporaneous preparation binding. See T018E_review_historical_audit.json and T018E_review_fix.md. Six focused tests pass, including rejection of mapping replacement plus prepared-hash refresh; an independent-verifier copied-fixture integration passes the bound case and rejects the changed case. The copied fixture is test-only and is not historical evidence.


The immutable T018-D selector passes4/5 predeclared clauses on one fresh40-image/120-episode cohort. Pooled improvement transfers, but quadrants safety fails. This is a fresh qualification negative under the accepted rule. No refit, threshold change, alternate cohort, second pass, or follow-on experiment was performed.

## Frozen protocol and provenance

- Research task: main0aa6d639; accepted D merge9e3a244709e64915ca30fe70335b60a42951bfa6.
- Scientific source: f0403a344233c0f082d7cf9a6c7f54ba63011efb. Runtime verification checks77 scientific/helper/lock files against real Git objects before each phase. T016-A/B renderer and feature bytes are unchanged; donor source7352c073a0d34af3aee63b6817fa02b63f79f860. Configuration is research_log/T018E_pipeline_lock.json.
- Frozen D receipt: db194f4caa897094655a36523fa074772d8cd7302ffda250a72e8b81cc2f9d94; x head0efaab934acc7ca2988b427e04def6116e266ab02e0e87487aa07a75728b21b4; y head212ce052f6c0409a7c02bd4fdcf56eb3105274b6ae4dbf2761ec4bd60c8e46b4. Weights, normalization, class order, schema, and inference source are exact accepted bytes.
- Historical metadata audit includes30 manifests. Exclusions contain705 IDs:688 used source images and17 previously inspected short images that fail original eligibility anyway. All T016–T018 development IDs are included. Full bindings are in T018E_exclusions.json; independent verification rehashes every recorded manifest using current committed Git blobs and reconstructs the union.
- The cohort is exactly the first40 eligible ascending numeric IDs in the existing official COCO val2017 pool after those exclusions; min(width,height)>=320 unchanged. Manifest SHA2569f3cff5bb06ed0b3466448efb59cc0b57228fdd777151339cbaee3e21524efd0. Each source appears once in each original left/right, quadrants, and40%-offset family. No alternative cohort was inspected.
- Canonical corner values come from the unchanged frozen Sobolev Region2 trajectory: fresh identity/Adam0.03 per episode, up to40 projected updates, fixed original gate, minimum predicted energy with earliest checkpoint tie-breaking. All frozen feature/head parameters retain no gradients. Five hard-cross renders only for direction inputs, tau0, original28-D schema. The scalar energy is used only by the pre-existing canonical trajectory, never as a direction fallback.

## Information boundary and GPU execution

Manifest bookkeeping and offline corruption synthesis are separate processes. Offline synthesis necessarily reads source pixels to create degraded-only tensors after the manifest freezes; it computes no metrics and loads no learned model. The selection process receives only opaque row-indexed degraded tensors, numerical gate constants, frozen assets and code/hash receipts. It cannot open the source JPEGs, cohort manifest/mapping, targets or evaluation tables. No family/ID/mask/gain value is a feature or branch input. No reference gradients/Jacobians or reference MSE enter adaptation or direction inference.

A6000 physicalGPU1/logicalcuda0 computes the accepted CLIP features and Region2 adaptation with seed7, TF32disabled. The same GPU handles post-freeze rendering; metric reduction follows the original CPU MSE implementation. Feature run20260913-131914-ttie-t018e-fresh-features exits0; separate evaluation run20260913-132546-ttie-t018e-postfreeze-eval exits0. GPU feature phase takes about265seconds.

Before selecting the fresh cohort, the existing D development replay found a remote PyTorch2.4.0+cu121 CPU logit difference of at most2.384185791015625e-6 versus the original Windows PyTorch2.13 CPU (zero class changes). To preserve exact frozen inference behavior, all120 GPU feature rows were transferred to the original Windows CPU backend and passed as one batch through the unchanged D API. This is a process split, with no model or feature setting change. Original development replay, new wrapper integration and independent replay all match exactly. The tiny direction heads remain CPU; all substantive feature/adaptation work uses the A6000.

| Event | UTC on2026-09-13 |
|---|---|
| Unique manifest frozen | 05:19:21.646699 |
| Offline degraded inputs completed | 05:19:24.683553 |
| GPU feature extraction began | 05:19:30.845806 |
| All120 feature rows frozen | 05:23:55.667819 |
| All120 direction decisions frozen | 05:25:05.470563 |
| Independent reference-free exact replay passed | 05:25:08.894166 |
| Reference evaluator first opened mapping/source targets | 05:25:53.980106 |
| Evaluation completed | 05:25:57.082450 |

Feature SHA256743124a38139597a20cf9aa9dc0f9ae0c972b6b95b34c66e0bacb8e193d4eb98. Decision SHA256c7aa98a8c3ce8c1042a712a42a3445a4e395755f5986d703646211a4476c757a. Decision-freeze receipt SHA256d389dec37629ce56a33884c47a0c9f11ce16c6ba2a2cec02c41f0203e323e201. Both processes record no reference access/blocked attempts. The reference evaluator requires the completed120-row freeze and checks its bytes before opening reference data.

## Exact results

H0 is the canonical(0.5,0.5,0) renderer. H1 uses the frozen predicted boundary. H* is the minimum of all nine hard candidates, computed only after global decision freeze. These are means of per-episode MSE; comparisons below use full precision with no rounding relaxation.

| Group | Episodes | H0 | H1 | H* | H1/H0 | H1/H* | Beneficial/equal/harmful |
|---|---:|---:|---:|---:|---:|---:|---|
| spatial_pool | 120 | 0.032644963526399805 | 0.030434589117066934 | 0.02984076495243547 | 0.9322904922976755 | 1.019899763480527 | 61/46/13 |
| left_right | 40 | 0.030260979582089932 | 0.02920097572496161 | 0.02848928513703868 | 0.9649712642562407 | 1.0249809914323778 | 26/6/8 |
| quadrants | 40 | 0.02911204905831255 | 0.029455147736007346 | 0.029108032450312748 | 1.0117854527177925 | 1.011925068665741 | 0/37/3 |
| offset_left_right_40 | 40 | 0.03856186193879694 | 0.03264764389023185 | 0.03192497726995498 | 0.8466303816462032 | 1.0226364020298608 | 35/3/2 |

| Clause | Observed ratio | Required maximum | Result |
|---|---:|---:|---|
| Pooled gain |0.9322904922976755|0.97|PASS|
| Pooled oracle proximity |1.019899763480527|1.03|PASS|
| Offset gain |0.8466303816462032|0.95|PASS|
| Left/right safety |0.9649712642562407|1.01|PASS|
| Quadrants safety |1.0117854527177925|1.01|FAIL|

Pooled MSE decreases6.77095077023245%; offset decreases15.33696183537968%; left/right decreases3.50287357437593%. Quadrants MSE increases1.17854527177925%, exceeding the1% limit by0.17854527177925 percentage points. Quadrants has37 unchanged and3 harmful episodes, with no beneficial episode. Those three harmful choices move only bx: image61108→(0.6,0.5),61471→(0.6,0.5),65798→(0.4,0.5); canonical is already the nine-hard oracle for each of these three. This localizes the observed failure to unnecessary x-boundary moves in these cases; it does not justify a post-hoc family rule or confidence threshold.

## All harmful cases, opened only after decision freeze

| Row | Source ID | Family | bx/by | H0 | H1 | H1/H0 |
|---:|---:|---|---|---:|---:|---:|
| 3 | 60932 | left_right | 0.4/0.4 | 0.012027909979224205 | 0.012933090329170227 | 1.0752566615072394 |
| 6 | 61108 | left_right | 0.4/0.5 | 0.04497929662466049 | 0.04657704383134842 | 1.0355218361910075 |
| 7 | 61108 | quadrants | 0.6/0.5 | 0.04630844295024872 | 0.04919250309467316 | 1.0622793590258026 |
| 14 | 61333 | offset_left_right_40 | 0.4/0.4 | 0.06585762649774551 | 0.07017801702022552 | 1.065601977359872 |
| 19 | 61471 | quadrants | 0.6/0.5 | 0.04405994713306427 | 0.05405443161725998 | 1.226838322207052 |
| 23 | 61584 | offset_left_right_40 | 0.4/0.6 | 0.0642121285200119 | 0.06532921642065048 | 1.017396836491574 |
| 30 | 61960 | left_right | 0.4/0.4 | 0.029121512547135353 | 0.03422289714217186 | 1.1751758115853197 |
| 36 | 62353 | left_right | 0.4/0.4 | 0.039783019572496414 | 0.044348254799842834 | 1.1147533615196608 |
| 42 | 62554 | left_right | 0.5/0.4 | 0.009805385023355484 | 0.009872057475149632 | 1.0067995750942305 |
| 87 | 64868 | left_right | 0.5/0.6 | 0.0116968285292387 | 0.011912570334970951 | 1.0184444702419087 |
| 99 | 65485 | left_right | 0.5/0.6 | 0.04169977456331253 | 0.04270286113023758 | 1.0240549637840863 |
| 105 | 65798 | left_right | 0.4/0.5 | 0.018018601462244987 | 0.02136346511542797 | 1.1856339217109382 |
| 106 | 65798 | quadrants | 0.4/0.5 | 0.03251582756638527 | 0.03336123004555702 | 1.0259997220567663 |

## Validation and artifacts

Five focused tests PASS: exact donor feature equivalence, deterministic eligible first40 selection, reference/mapping access rejection, mandatory global freeze before reference evaluation, and literal gate/harmful-case arithmetic. Four existing direction-selector lifecycle tests PASS. Original D120 development logits replay exactly on the original CPU backend. The new local wrapper and independent verifier also exactly reproduce the same development fixture; synthetic unit-test fits never touch final heads.

Independent verifier phaseA imports no TTIE implementation, uses only frozen five-cross features plus the D receipt/heads, reconstructs the84-D inputs and network arithmetic from saved normalization buffers, and exactly reproduces all120 fresh logits/classes/boundaries/indices without reference access or normalization refit. PhaseB independently rebuilds provenance/disjointness, verifies global timing/hashes, selects H0/H1/H* from each frozen nine-MSE table, and recomputes all120 metrics, aggregates, ratios, outcomes and five booleans. This is independent table arithmetic, not a second pixel or feature run. Receipts: T018E_replay_verification.json and T018E_metric_verification.json.

Compact run artifacts are committed under research_log/remote_runs/20260913-131914-ttie-t018e-fresh-features/ and the separate evaluation-run directory. They include manifests, input hashes, all120 corner states/trajectories, features/decisions/freeze receipts, per-episode nine-MSE tables, summaries and logs. The120 degraded tensors stay remotely, with identical500-file/214040782-byte run copies verified under both /home/wenchang/asdasdsad/wjq/TTIE/runs and /media/wenchang/F/wjq/TTIE/runs. T018E_remote_mirror.json is the full file/hash/byte manifest. The fetched compact archive is retained at .autodl/T018E_complete_small.tar.gz, SHA256b4914cde6d74d7057bfcb3a8dea274e88b065e77314d5d979d9192c7b8918868.

For review, run the two phases of research_log/T018E_verify.py with the saved selected/cohort/evaluation directories and the original frozen D artifacts. Do not relaunch cohort construction or adaptation to review the saved result. Source/weights/features/decisions and original data hashes are preserved for recovery.

The task is complete with a fresh negative. Wait for the research lead to review this evidence and issue a new task; no follow-on tuning or experiment is authorized by this result.
