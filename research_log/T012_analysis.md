# T012 Stage A — completed negative result

2026-09-12T07:44:00Z. The learned trajectory selector passes **1/5** predeclared source clauses. Stage B was not opened: no fresh T012 manifest was created or read, and no fresh evaluation images were scored. No tuning or rerun followed this result.

## Reproducible experiment

- Frozen source and source manifest: `61b7eb6875343a887b2e55afc96819bab51d60ab`; implementation `2d795aefb79c6ea40c78e7e50bcc83f8ba2f8067`.
- Branch: `codex/T012-learned-trajectory-stop`.
- A6000 release: `20260912-151440-ttie-t012-stage-a`; run: `20260912-151505-ttie-t012-stage-a`; exited 0 at `2026-09-12T07:30:42Z`.
- Command: `CUDA_VISIBLE_DEVICES=1 bash scripts/run_t012_a6000.sh A 61b7eb6875343a887b2e55afc96819bab51d60ab`. Physical second A6000; unrelated first-GPU job preserved.
- Official COCO image files only, 80 train images and 20 calibration images, five conditions each, excluding all 308 prior IDs. Source manifest SHA256: `bcd0f7ca780182dab34e6bc2f83cd50e65a58458c689edbc49eb2b5b837ff6f7`. All 100 now remain development data; future decisive exclusions total 408 IDs.
- Full frozen T011 projected Region2 trajectory, unchanged semantic/projected generator files. A passive recorder captures the actual scored images, gradients/diagnostics and checkpoint states. The head selects an existing checkpoint; it does not shorten generation or alter updates.
- Specified 33 features; MLP 33→64→64→1, ReLU, Huber delta 1, AdamW lr .001 / weight decay .0001, batch 256, seed 7, exactly 100 epochs, final weights only. Train-only population normalization; canonical CPU one-thread training and selection.
- 10,493 training checkpoint rows from 80 images; 3,100 calibration checkpoint rows from 20 images. Training Huber loss: epoch 1 `.21873782635910954`, epoch 100 `.005240807232461968`; no epoch selection.
- Head SHA256: `cfcff2c20c04225f0b4300d658ee245eb00456f194154733e0944f5fa7aed17b`. Failed source receipt and checkpoint remain under the actual run audit directory; neither is promoted as an approved Stage-B asset.

## Frozen source qualification

Each condition has 20 images; pooled heterogeneous means average all 40 LR/quadrant episodes. Ratios are ratios of pooled mean MSE, not means of per-image ratios.

| Clause | Observed | Required | Result |
|---|---:|---:|---|
| Clean p95 MSE | .005111466511152687 | <=.005 | Fail |
| Dark MSE / identity | .6574603090545682 | <=.65 | Fail |
| Bright MSE / identity | .4033348411132593 | <=.65 | Pass |
| Heterogeneous MSE / matched discrete | 1.0977345390901487 | <=.95 | Fail |
| Heterogeneous MSE / source fixed step | 1.138455355161012 | <=.95 | Fail |

Report-only learned/oracle heterogeneous regret ratio: **1.1543505055659473**. The learned head has 9.77% higher heterogeneous MSE than discrete, 13.85% higher than fixed 16 and 15.44% higher than the checkpoint oracle.

| Method | Clean mean MSE | Clean p95 MSE | Dark MSE | Bright MSE | Heterogeneous MSE |
|---|---:|---:|---:|---:|---:|
| Identity | 0 | 0 | .073152555 | .045504754 | .059825797 |
| Region2 direct | .002744979 | .016906760 | .049792175 | .013654373 | .031376832 |
| Region2 projected discrete | .001059633 | .004942643 | .041348616 | .018221054 | .028861856 |
| Original projected final | .000792457 | .004028250 | .041399476 | .017751741 | .028984217 |
| Projected one step | .000062106 | .000412377 | .068604552 | .037643648 | .053490828 |
| Learned stop | .000734536 | .005111467 | .048094901 | .018353653 | .031682656 |
| Source fixed 16 | .001070099 | .004781726 | .040829941 | .016392433 | .027829512 |
| Reference-only checkpoint oracle | ~0 | ~0 | .040627462 | .015324100 | .027446305 |

The exact machine-readable aggregates, all per-case MSE/PSNR and reference-region metrics are in the run's `summary.json`, `summary.md`, `calibration_metrics.json` and episode files. Active step-0 rendering can have floating-point drift (~1e-18 MSE); all-inactive paths preserve exact identity.

## Source fixed-depth choice

The prescribed clean-p95 eligibility followed by minimum heterogeneous MSE selects **16**. Candidate indices clamp to the last saved checkpoint after the original semantic stop; no additional updates are generated.

| Fixed candidate | Clean p95 | Heterogeneous MSE | Clean eligible |
|---|---:|---:|---|
| 0 | 1.9840400470267e-17 | .05982579411938786 | Yes |
| 1 | .00041237663826905203 | .05349082823377103 | Yes |
| 2 | .0016213559545576574 | .04815330270212144 | Yes |
| 4 | .004727241257205606 | .04002553424797952 | Yes |
| 8 | .005074933962896469 | .030468204396311194 | No |
| 16 | .00478172632865608 | .02782951153931208 | Yes |
| 40 | .004028249625116587 | .028984217473771424 | Yes |

## Oracle bound and interpretation

This calibration result exposes both a selector error and a limitation of the available trajectory checkpoints. Per-image reference-MSE minima give a lower bound on the pooled MSE achievable by **any** selector restricted to these same saved checkpoints. The oracle heterogeneous MSE is `.027446305338526145`:

- Oracle / fixed 16 = **.9862302218188553**, only **1.37698%** improvement; required improvement is 5%.
- Oracle / discrete = **.9509542671807111**, only **4.90457%** improvement; this also misses the strict 5% clause. Do not round this into a pass.

Thus even perfect checkpoint selection cannot pass either heterogeneous improvement clause on this calibration set. Improving the head alone cannot meet the complete T012 Stage-A contract with these frozen trajectories. Separately, learned/oracle regret of 1.15435 confirms the learned ranking still leaves achievable restoration unused. These are descriptive, reference-only deductions after the frozen run; they did not change criteria or decisions. They do not prove that learned stopping generally fails, and they do not establish held-out performance.

Recommendation to the research lead: accept the controlled source-stage negative result and reconsider the available correction trajectory / task alignment in an explicit next task. Do not tune the inspected calibration set, relax the failed margins, or start a new inner objective automatically. No detector, meta-learning or ViT3 work has begun.

## Selected-step histograms

Each mapping is `step: number of images`; 20 per condition. Step 0 includes original no-active bypasses and may also include learned selections, so its count alone is not a head-specific rejection rate.

- Clean: `{0:17, 5:1, 14:1, 39:1}`.
- Dark: `{0:7, 7:1, 9:1, 12:4, 13:2, 14:2, 16:1, 36:1, 39:1}`.
- Bright: `{0:4, 6:1, 9:1, 11:1, 13:1, 15:2, 16:2, 26:1, 27:1, 28:1, 40:5}`.
- Left/right: `{0:2, 3:2, 8:2, 9:1, 11:1, 12:2, 13:2, 14:1, 15:1, 16:1, 40:5}`.
- Quadrants: `{0:4, 1:1, 8:1, 9:1, 12:3, 13:2, 15:1, 18:2, 23:1, 28:1, 31:1, 40:2}`.

## Verification and artifact location

Baseline 93 tests pass (26.953 s). Four implementation increments passed their focused tests; full suite **105 local tests pass (31.555 s)** and **105 A6000 tests pass (8.892 s)**. Original frozen calibration scores match bitwise. Tests cover CPU T011 trajectory/output/gradient parity, feature definitions, reference/metadata independence, train-only normalization, fixed recipe, earliest ties, all-inactive identity, staged source stopping, receipt checks and CPU/CUDA head fixture decisions. No scientific code changed after frozen source `61b7eb6`.

Actual remote audit verifies all **500 inputs / 13,593 checkpoints / 13,093 trajectory updates**, including 107 no-active inputs, **2,200 file hashes**, finite bounded outputs, identity reset, checkpoint/output linkage, exact feature recomputation, exact train-only normalization and exact frozen-head score/selection replay. Source summary and fixed-step selection recompute exactly. Local verification confirms **1,100 small-file hashes**, the head hash, all 800 calibration rows and all summary/fixed-step aggregates with maximum absolute difference **0.0**.

Five fixed first-calibration-ID **21167** panels were visually inspected: labels and all nine panels are intact; they are illustrative, not selected for favorable results. No additional images were chosen for figures.

Local tracked evidence: `research_log/remote_runs/20260912-151505-ttie-t012-stage-a/`, including head, features/scores/grids/raw states, per-checkpoint metrics, decisions, selections, hashes, failed receipt, manifests, training history, summary, five figures, tests/environment/run logs and both verification receipts. Tar SHA256: `7413cf328478d5925bb3ad6c6b71c59ed108e69b37ee67553a9e176858c53b8c`, exact match remote/local before extraction.

Full float32 image packs remain at `/home/wenchang/asdasdsad/wjq/TTIE/runs/20260912-151505-ttie-t012-stage-a/artifacts/audit/episodes/`: checkpoint packs 23,203,340,120 bytes; baseline output packs 24,392,544,072 bytes; calibration selected output packs 5,152,194,564 bytes; total **52,748,078,756 bytes**. Every file's relative path, byte count and SHA256 is preserved in the fetched episode receipts and verified remotely. Only these three large pack types were excluded from the receipt archive. Tested release remains unchanged; recovery notes are mirrored separately under the remote project root.

Failures/deviations: no failed software tests, experiment runtime failures, scientific deviations, repeated experiment or post-calibration tuning. The pre-existing NVML warning and CUDA bicubic backward nondeterminism remain disclosed; actual CUDA worked, and no strict bitwise CUDA trajectory-repeat claim is made. Canonical CPU selector replay is exact. Stage B is implemented and fixture-tested but has **no real-data result**, as required after failed Stage A.
