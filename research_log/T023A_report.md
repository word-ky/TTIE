# T023-A completion report

Status: experiment-complete. The one predeclared16-pair real-source Sobolev pilot fails both validation acceptance conditions: mean paired PSNR +0.2870812278742196 dB (required >=+0.50), mean paired SSIM -0.01374216598567974 (required nondecrease). No dataset scaling, second head, sweep or further tuning was run on the real data.

## Fixed100-image validation result

| Configuration | PSNR mean | PSNR median | SSIM mean | SSIM median |
|---|---:|---:|---:|---:|
| raw | 8.109722671657906 | 7.600161540858166 | 0.1600228434803769 | 0.13897726549987016 |
| T022C | 10.229554025363404 | 9.650373829782268 | 0.3282314776612914 | 0.3002515375673881 |
| T023A | 10.516635253237624 | 9.988665452257262 | 0.3144893116756116 | 0.2752833186601146 |

| T023-A minus C | Mean | Median | p10 | p90 |
|---|---:|---:|---:|---:|
| psnr | 0.2870812278742196 | 0.4726580338175159 | -1.6169762293712846 | 1.8704829673209742 |
| ssim | -0.01374216598567974 | -0.005937689068285887 | -0.09378033426264573 | 0.05794830019848839 |

Selected-step counts (nonzero): 12:11, 15:1, 25:1, 27:1, 28:1, 29:1, 30:1, 33:1, 34:2, 38:1, 40:79. Full0..40 histogram in comparison.json;79/100 select40. Runtime seconds mean2.4098959528136765, median2.445311989518814, p95 2.4736445495480437. All100 active,4000 updates total. Extra inherited final active/inactive saturation diagnostics are retained in comparison.json, without changing inference.

## Source selection and actual training

Exactly16 pairs selected before source normal-pixel decoding from the589 non-validation official-training pairs by ascending SHA256 of UTF-8 relative low path, tie by path. Ordered paths, low/normal byte hashes and disjointness are fixed in T023A_source_manifest.json. The original100 validation split SHA256 remains b88c8347005984b5523b117b52c0c068672fe172eb7c9aa5b60102d350e2d85b. Only those16 source pairs were extracted; official test was never decoded or used. Filename/header/hash work is distinct from pixel decoding.

For every source pair, the accepted C trajectory with the old T014 head ran exactly40 updates. All41 states/features, scores/grids, decisions and selected output were saved and hash-frozen before that pair's normal image was opened. There are656 source-supervised states. Low/source-normal decoder access is restricted to the declared source paths and its complete ordered audit matches the16 pairs; no validation normal paths enter bank construction or training.

The existing source_derivatives/derivative_record replays these EV2 raw states with the same Region2 renderer and frozen gate. It does not call the old synthetic source_bank and does not apply the old +0.5 projection. Projection is not differentiated, matching T014. The28x8 Jacobian has zero first12 constant rows and full CLIP/grid derivatives in the remaining rows. A focused CPU/A6000 test at EV1.2 (>old+.5 cap) verifies cached J-transpose energy gradients against direct autograd, and exact reference-gradient equality. Real656 cached features match the derivative path.

Exactly one new EnergyHead was trained from scratch:28->[64,64]->1 SiLU, value Huber plus existing cosine derivative loss [1,1], AdamW1e-3/weight decay1e-4, batch256, seed7,100 epochs, final epoch only. The unchanged T014 train_head supplies normalization and training, with the exact primary derivative callback; no real-data value-only control was trained. The historical recipe's CPU device is retained for this tiny head; all source image/CLIP trajectories and derivatives, and validation adaptation, ran on A6000 physicalGPU1. Synthetic recipe tests may instantiate comparison heads but never use real source/validation data.

## Old versus new source-bank diagnostics

These are training-bank statistics, not validation evidence.

| Metric | Old T014 | New T023-A |
|---|---:|---:|
| Direction rows |656|656|
| Positive cosine fraction |0.667682945728302|1.0|
| Median cosine |0.3177952170372009|0.9881367683410645|
| Direction loss |0.3908548653125763|0.01679924689233303|
| Standardized value Huber |1.00216543674469|0.015469797886908054|
| Unstandardized log-MSE RMSE |2.323836876197185|0.13799735554402304|
| Unstandardized log-MSE MAE |1.9987316275548417|0.07207272249498649|

Standardized Huber uses each head's own saved normalization; the unstandardized errors provide a common value scale. Full per-row cosines and100-epoch history are saved. Strong source value/gradient fitting did not yield the required joint PSNR/SSIM validation improvement. This pilot does not establish that larger training will succeed or fail; any next strategy is for research-lead review. Do not promote this checkpoint over C based on PSNR alone.

## Freeze order and independent checks

New checkpoint frozen2026-09-13T22:17:55.658000+00:00, SHA256 e9f894a3c211f529bc2015375198f376fea4c7125086ec2de0a058e7c81f1770. Training receipt SHA256 1da57f451e767276bfffaca987d52fe2471f152f5158602aa00f8491c28738d7; verified22:18:30.562893Z before validation launch. Validation changes only the energy asset. AST-equivalent C runner retains EV2/bright/gamma bounds, gate, CLIP/prototypes, identity initialization, Adam.03,40 updates and minimum learned-energy selection.

All100 validation outputs/decisions/trajectories frozen2026-09-13T22:23:05.732907+00:00, SHA256 458f63f15ebbe6d80a70516d7e99921a87de7c36e1ab67af5dc428a003c5f21f. Validation normal-reference deployment started22:23:33.074224Z, strictly afterward. No validation normal-root argument in inference; its low-only decoder recorded exactly100 low opens. Changing/withholding100 synthetic validation targets preserves all inference artifact hashes. Actual validation ran once only.

Exact accepted evaluate_t022a.py reused unchanged: full-frame RGB float32 input/float64 PSNR, Gaussian11 sigma1.5 RGB SSIM, no crop/resize. All metrics finite; independent metric maximum error7.105427357601002e-15. All100 gates/action boxes match C; raw metrics match C; all scientific artifact hashes remain unchanged after references. Independent NumPy/statistics aggregates agree. Local200 decision/trajectory hashes and paired aggregation/histogram pass; independent old/new source statistics max difference1.1920928955078125e-07 across CPU environments. New checkpoint/training receipt remained unchanged and preceded validation.

## Tests, provenance, artifacts and stop

Baseline C tests3 pass14.88s. New focused tests5 pass11.70s: deterministic source selection, validation-runner equivalence, matched EV2 derivatives, exact original primary training recipe, and100-target mutation/withholding. Same5 tests pass remotely, including the CUDA derivative case. Both source and validation jobs exit0. No scientific failure, rerun or deviation; existing NVML/open_clip loader warnings did not affect execution.

Scientific source commit b8309795231033e97359489ab87e8802e00f1589; accepted task mainf674f4a1f16718e7d7270eb6e182902285fa2faf. PR https://github.com/word-ky/TTIE/pull/46. Source job20260914-061503-ttie-t023a-source; validation job20260914-061849-ttie-t023a-validation. Exact launch commands/configs, source/asset/split hashes, source freeze/supervision receipts, new energy.pt, training history, source-value diagnostics, validation metrics/paired deltas and independent audits are under research_log/remote_runs for those two jobs. Local source/data/proof/test/deployment/delivery receipts are research_log/T023A_*. Evaluation commands are T023A_evaluate.sh. Original output.pt tensors remain under /home/wenchang/asdasdsad/wjq/TTIE/runs.

Full execution backup /media/wenchang/F/wjq/TTIE/shared/t023a/T023A_execution.tar SHA256 0a806c89d5a0bcb172b021d7050b171c0075044dd1410beb589f88b386af823d. Compact transfer SHA256 9e6689bd4afa77f63307b604fe9c1894b6c09bd80a5c069448df4ec83f0afdcb. No official-test/SOTA claim, full589-pair scaling, external baseline or additional tuning. PR46 awaits research-lead review; no self-merge. Stop after this pilot.

negative/insufficient