# T022-D completion report

Status: experiment-complete. Extending the accepted T022-C trajectory from40 to80 updates fails both predeclared criteria: mean paired PSNR -0.013835819555952042 dB (required >=+0.50) and mean paired SSIM -0.0016948707771770898 (required nondecrease). No additional budget or tuning experiment was launched.

## Results on the frozen100-image LOL-v2 validation split

| Configuration | PSNR mean | PSNR median | SSIM mean | SSIM median |
|---|---:|---:|---:|---:|
| raw | 8.109722671657906 | 7.600161540858166 | 0.1600228434803769 | 0.13897726549987016 |
| T022C | 10.229554025363404 | 9.650373829782268 | 0.3282314776612914 | 0.3002515375673881 |
| T022D | 10.215718205807452 | 9.7207020079187 | 0.3265366068841143 | 0.29904259290902413 |

| D minus C | Mean | Median | p10 | p90 |
|---|---:|---:|---:|---:|
| psnr | -0.013835819555952042 | 0.01642880813071468 | -0.19940420893637753 | 0.2214177233970368 |
| ssim | -0.0016948707771770898 | -0.0008018400800252068 | -0.010335891635346562 | 0.009320225728334984 |

82/100 images select step80. Nonzero selected-step counts: 61:1, 68:1, 69:2, 70:1, 72:2, 73:2, 74:2, 75:3, 76:2, 79:2, 80:82. Full0..80 histogram is in comparison.json. More optimization of the learned energy did not improve reference restoration quality in this fixed comparison; terminal-checkpoint selection alone does not establish that extending the budget improves PSNR/SSIM. The run does not support replacing the40-step C candidate. The next research-lead review decides further work.

Runtime seconds: mean4.554767821141286, median4.68278713800828, p95 4.786519789206795. C mean2.431485087046749. This is one fixed comparison, not an efficiency sweep.

## Final-state saturation, with inactive coordinates separated

Saturation uses absolute distance<=1e-6 from the frozen box, at final step80. Across100 images there are392 active and8 inactive spatial coordinates per channel. All8 inactive coordinates have collapsed bounds; all392 active coordinates have noncollapsed bounds. Inactive and collapsed refer to the same8 coordinates and must not be added twice.

| Channel/group | Coordinates | Lower hits | Upper hits | Either hits | Either fraction |
|---|---:|---:|---:|---:|---:|
| ev_active | 392 | 22 | 11 | 33 | 0.08418367346938775 |
| ev_inactive | 8 | 8 | 8 | 8 | 1.0 |
| gamma_active | 392 | 272 | 57 | 329 | 0.8392857142857143 |
| gamma_inactive | 8 | 8 | 8 | 8 | 1.0 |

Detailed lower/upper fractions and collapsed/active-noncollapsed accounting are in comparison.json. Selected-state saturation across all400 coordinates is additionally preserved there and in paired_deltas.csv; it is distinct from this final-step table. These diagnostics were computed after freeze from existing gate, box and final-grid metadata, without changing inference.

## Implementation and unchanged scientific inputs

Scientific source commit ffa460bf4a2f29819fe5334b7f785f6952d3603c; PR https://github.com/word-ky/TTIE/pull/45. Accepted C merge1d4818b37fdab8f6e3348eae39da9f8c0382858f; task main eb7f1da1cc48e6b98949741153e8febef7988437. Explicit ttie.lolv2_budget_core runner copies accepted C with only task labeling and max_steps40->80 changed. It calls the unchanged accepted C trajectory. Historical C/T014 source and artifacts are not edited. Bounds remain active dark[0,2], bright[-.5,0], inactive identity, gamma[.8,1.25]. Same frozen CLIP/head/prototypes/gate, Region2 renderer, identity initialization, Adam lr.03, and minimum predicted-energy checkpoint rule.

Split SHA256 b88c8347005984b5523b117b52c0c068672fe172eb7c9aa5b60102d350e2d85b; same100 ordered native600x400 validation images. Asset identities are recorded in T022A_assets.json and runtime config.json and checked before/after inference. All100 episodes ran once on A6000 physicalGPU1 and completed80 updates, total8000. No official-test decoding/inference/scoring, sweep, retraining, baseline execution or LPIPS.

## Target isolation, metric checks, and observed operational failures

No normal-root argument in inference. The accepted low-only PIL decoder whitelist logs all100 opened low images. A synthetic100-reference counterfactual test with changed/withheld targets confirms identical80-step output/decision/trajectory hashes; actual validation images were not rerun for this test. CPU/GPU exactEV endpoint and finite-gradient checks pass. Local initial C/D tests5 pass19.86s; complete D tests4 pass13.55s. All4 D tests and the retained endpoint test also pass on the A6000 host.

All outputs/decisions/trajectories frozen2026-09-13T21:46:11.044221+00:00; freeze SHA256 b86500512fdf415621a36b6f72509cd30476252276f72c61c8dc58fabb1f2ef0. Evaluation reference deployment started2026-09-13T21:46:43.499719+00:00, strictly afterward. Metric audit completed2026-09-13T21:47:27.238831+00:00. All100 gates/action boxes equal C; all scientific artifact hashes survive post-reference evaluation. All outputs/metrics finite, raw metrics exactly C, selected indices are minimum learned energy.

Metric arithmetic remains exactly the accepted full-frame RGB float32-input/float64 PSNR and Gaussian11 sigma1.5 RGB SSIM. evaluate_t022d.py changes only the evaluator's expected-update assertion40->80; an AST test checks that the rest equals accepted evaluate_t022a.py. Independent Torch PSNR and explicit separable Gaussian SSIM maximum absolute error5.329070518200751e-15. NumPy/statistics comparison agrees; local standard-library reaggregation, histogram, active/inactive saturation counts, and200 decision/trajectory hashes independently pass.

One SSH status read timed out and recovered on retry; the remote job continued. Windows CRLF in the evaluation shell prevented its initial cd/redirection after the successful post-freeze reference copy. Normalizing shell line endings and launching only the evaluator/comparison resolved it, preserving the original freeze/deployment receipt; no inference rerun or metric arithmetic change. Known NVML initialization warning did not prevent CUDA use.

## Artifacts, reproduction, and stop

Run20260914-053820-ttie-t022d-80steps under /home/wenchang/asdasdsad/wjq/TTIE/runs. Exact GPU launch is recorded in run.sh/config.json; post-freeze evaluation commands in research_log/T022D_evaluate.sh. Source/proof/diff, fixed evaluation binding, tests, freeze/deployment receipts, local audit and delivery live under research_log/T022D_*. Compact run evidence includes100 decisions/trajectories, opened-image audit, per-image metrics and paired deltas, full histogram, saturation counts, independent verification and logs. Original selected output.pt files remain on the server.

Full backup: /media/wenchang/F/wjq/TTIE/shared/t022d/T022D_execution.tar, SHA256 c39a943953eae3272cb16683f979eb56e48b657d89afb47c491c69df979c4559. Compact transfer SHA256 fc666fe3a716053b0a0f1dc8188d0fd204b3ca2dad402fba8345c5296d9ea5d0. PR45 awaits research-lead review; no self-merge. T022-D stops here. Validation only; no official-test/SOTA claim.

negative/insufficient