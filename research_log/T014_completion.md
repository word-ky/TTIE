# T014 completion report

T014 implements a source-supervised Sobolev restoration energy while holding the T013 feature representation, MLP, source state bank, training budget and projected test-time optimization fixed. The same-source value-only head is the matched control. The accepted source-cache numerical repair preserves the original feature-consistency tolerance and all scientific settings.

The result supports the derivative-supervision hypothesis under this fixed synthetic restoration protocol. Stage A passes all eight development clauses; the frozen Stage B passes all twelve fresh clauses. Source Huber fit is worse for Sobolev, yet held-out calibration gradient alignment and fresh restoration are better. No inference step consumes clean references, labels, degradation metadata, source Jacobians or reference gradients. Reference metrics and the oracle are computed after label-free outputs are persisted.

Scientific source: `f861b2c6ffde6d017cb174ef8e00cb75701bf5e1`. Engineering branch: `codex/T014-sobolev-energy`. PR: https://github.com/word-ky/TTIE/pull/14. Complete evidence commit and publication receipt will be recorded in the main Codex mailbox.

## Fresh qualification: literal conjunction

| Clause | Observed | Threshold | Result |
|---|---:|---:|---|
| Clean mean MSE | 0.0002532600949052721 | <=0.003 | PASS |
| Clean p95 MSE | 0.00012560087488963485 | <=0.005 | PASS |
| Homogeneous dark / identity | 0.570950280201505 | <=0.60 | PASS |
| Homogeneous bright / identity | 0.4204195234145255 | <=0.60 | PASS |
| Heterogeneous / global Sobolev | 0.5624951651458492 | <=0.85 | PASS |
| Heterogeneous / direct | 0.846958576027204 | <=0.95 | PASS |
| Heterogeneous / projected discrete | 0.8827539376865203 | <=0.95 | PASS |
| Heterogeneous / fixed semantic16 | 0.9168763941227348 | <=0.95 | PASS |
| Heterogeneous / same-source value-only | 0.8082997257311235 | <=0.95 | PASS |
| Heterogeneous dark-region / identity | 0.5315698754000598 | <=1.05 | PASS |
| Heterogeneous bright-region / identity | 0.4476425847437993 | <=1.05 | PASS |
| Exact-quadrant / bilinear Sobolev | 0.7143678745065832 | <=0.90 | PASS |

Forty fresh images produce 200 primary inputs and 40 report-only offset inputs. Heterogeneous qualification uses 80 left-right/quadrant inputs. There is no fresh fitting, retuning, second split or rerun. All 648 inspected prior/source/fresh image IDs are now unavailable for corrective tuning.

Heterogeneous MSE is 0.03385802966658957: 19.17% below value-only, 11.72% below discrete and 8.31% below fixed16. The reference-only oracle is 0.03293453548103571; primary/oracle regret 1.0280402978838314, oracle/discrete 0.858676395763497, oracle/fixed16 0.8918681456457281. The oracle is not the inference selector.

## Development evidence kept separate

Both heads fit the same 7346 source states (7248 eligible directional rows). Value-only/Sobolev standardized Huber: 0.033193279057741165/0.051005665212869644; source median cosine: 0.46205344796180725/0.9724292755126953. These are training diagnostics only.

On 74 active non-clean held-out calibration episodes, positive cosines are 59/74 versus 73/74, and median cosine 0.4246446532217347 versus 0.9360590709945287. Sobolev-minus-control deltas are +0.18918918918918926 positive fraction and +0.511414417772794 median. Calibration heterogeneous Sobolev/value-only is 0.8342483020057921; all eight development clauses pass. Full distributions and source/calibration method tables remain in T014_analysis.md and Stage-A final_distributions.json/.md.

## Trajectories and limits

The frozen minimum-energy selector uses earliest ties among identity plus 40 active projected Adam updates. All 45 all-inactive fresh inputs select identity with zero updates. Across all 240 inputs, the primary projects 7395/7800 updates and ends with 642/1054 movable coordinates at a boundary. Complete per-condition selected-step histograms, primary-only and heterogeneous-only aggregates are in Stage-B final_distributions.json/.md. The supported mechanism includes the frozen projected action geometry; this is not evidence for an unconstrained learned energy.

Offset stress remains report-only: primary MSE 0.044768725894 versus bilinear 0.044506553258, ratio 1.0058906524407158. Primary/discrete is 0.9767600741572411 and primary/fixed16 0.9818854118426412. Hard Region2 does not resolve the boundary-misalignment limitation. Global Sobolev also remains better on homogeneous dark/bright. No claim of universal spatial superiority or full natural-degradation benchmark qualification is made.

Six fixed representative panels use first fresh image 50844. All ten methods and clean reference are present; visible incomplete recovery and boundary artifacts are retained. The clean mean/p95 clauses are not a worst-case guarantee: 38/40 clean inputs bypass, two are active.

## Reproducibility and execution

- Final scientific regression: 125 local tests pass in 135.151s; 125 repaired Stage-A A6000 tests pass in 37.822s; 125 Stage-B A6000 tests pass in 37.516s. Original calibration preflight is bitwise equal. No scientific code changed after the tested source SHA.
- Freeze receipt and both checkpoint blobs committed/verified at `6a9870f836b6763f843d78e5c549eba432b0a150`. Only then was fresh manifest `7416a1b91949985312cef002d47568091fb57761` generated/committed before scoring. Fresh manifest SHA256 `1e891b2d9049b0bc7f68fbc1e656f1e90c8cb88ff0d526fbe20cf950c293d8b5` excludes 608 prior/source IDs.
- Sobolev SHA256 `c3d1eef9f20af163e1268823cf16d3fdaed315e7723fc94db3b1138ad1336521`; value-only SHA256 `df6c5af2610e741cf03a59239135a82204550fab3bcbf9e408e553521ce7b69c`; both unchanged through calibration and fresh evaluation. Source receipt, CLIP, prototypes, T007 calibration and source/fresh manifests verified.
- Stage B run `20260912-191947-ttie-t014-stage-b`, release `20260912-191938-ttie-t014-stage-b`, physical GPU1, exit0 at 2026-09-12T12:00:42Z. Command: `env AUTODL_ARTIFACTS_DIR="/media/wenchang/F/wjq/TTIE/runs/$AUTODL_RUN_ID/artifacts" CUDA_VISIBLE_DEVICES=1 bash scripts/run_t014_a6000.sh B f861b2c6ffde6d017cb174ef8e00cb75701bf5e1`.
- Remote reporting-only verification: 5280 file hashes, all stored-pixel MSE, both GPU head energies and minimum-energy selections, feature algebra, projections and summary agree exactly. Counts: 240 inputs, 32160 energy checkpoints, 31200 energy updates, 6947 semantic checkpoints, 21106 exact inactive-region checks. No source alignment or fitting occurs in Stage B.
- Full 76,940,687,040 large image bytes remain in `/media/wenchang/F/wjq/TTIE/runs/20260912-191947-ttie-t014-stage-b/artifacts/audit`. Execution metadata/logs remain in `/home/wenchang/asdasdsad/wjq/TTIE/runs/20260912-191947-ttie-t014-stage-b` and are copied into the F run archive. All earlier packs remain untouched. The 38 MiB small archive excludes only the large pixel tensors and preserves all other evidence; SHA256 `8aa3419e786dfdca6c046204b9b052f0ef5ac51bfab37ae2c4282ded4710762b`.
- Accepted initial Stage-A CLIP cache failure, repaired same-mode forward, source rendering roundoff and inherited CUDA backward nondeterminism remain documented in T014.md/T014_analysis.md and preserved receipts. No outcome-dependent repair occurred. Direct Jacobian/autograd equivalence was verified on the fixed real-CLIP fixture, not rerun at every bank state; no stronger equivalence claim is made.

The next action is research-lead review of PR14 and this evidence. T014 stops after delivery; no T015, feature expansion, learned basis, detector, meta-initialization, prompt or ViT3 work is started. The existing 15-minute heartbeat continues reading GitHub and reports only meaningful changes.

## Completed local verification and final reporting

The completed archive download exited0 and its SHA256 matches the remote archive exactly. Local verification passes 2640 small-file hashes and all 2400 method rows; both frozen head hashes match. The recomputed summary differs by at most 5.551115123125783e-17 (floating-point rounding), with all twelve clauses unchanged. Reporting script summarize_t014_b.py completed from those verified records, producing full method tables, all/per-condition histograms, projection and movable-boundary counts, oracle and offset ratios. It changes no scientific output or gate.

Primary-only (200 inputs, excluding stress): 5900/6280 projected updates =0.9394904458598726; 542/864 movable final boundary coordinates =0.6273148148148148; 43 all-inactive inputs. Heterogeneous-only (80): 3042/3160 projected =0.9626582278481013; 288/436 movable final boundary =0.6605504587155964; one all-inactive input. All/all-condition/full-primary selected-step distributions are retained in final_distributions.json/.md. These denominators are reported separately from all240 including stress.

Local evidence root: research_log/remote_runs/20260912-191947-ttie-t014-stage-b/. Remote reporting-only audit log is T014_stage_b_verification.log in the same run. The source scientific code remains f861b2c; only reporting scripts/documents/evidence have been added since the final tested code. Full experiment and all requested verification/reporting are complete. Await research-lead review; no further task started.

Verification commands: python -m unittest discover -s tests -v; remote research_log/verify_t014_b.py from the frozen release with actual F audit and --images shared/t008/val2017; local research_log/verify_t014_b_local.py with --archive .autodl/T014_B_final_small.tar.gz; research_log/summarize_t014_b.py on the completed local B audit. Files changed include ttie/sobolev_source.py, sobolev_train.py, sobolev_metrics.py, sobolev_receipt.py, sobolev_io.py, sobolev_pilot.py, prepare_t014.py, reused energy/training/serialization adapters, focused tests and scripts/run_t014_a6000.sh; final delivery adds reporting scripts and complete verified receipts.
