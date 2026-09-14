# T025-A fixed reference-oracle diagnostic

Label: **REFERENCE_ORACLE_ONLY**. Research task main: `ee1c9ee9` (exact main commit in progress log). T024-A accepted merge: `48617186ea14f8394f4673c2e930112bf3d70f87`. Scientific implementation commit: `04e436dbefad747a562fd68faacecaf5d90006c9`.

This task measures states reachable by a fixed reference-assisted search inside the accepted T022-C action family. It is not a deployable algorithm, a new trained head, a SOTA comparison, or a certified global optimum. No official-test image is read.

## Fixed reuse and optimizer

- Exactly the100 frozen LOL-v2 Real validation pairs; split SHA256 `b88c8347005984b5523b117b52c0c068672fe172eb7c9aa5b60102d350e2d85b`.
- Reuse each image's T022-C saved decision, selected raw state and output from run `20260914-042925-ttie-t022c-ev2`. Bind decision/output/trajectory bytes to that run's freeze receipt before reference optimization. Do not call the low-only gate again.
- Use unchanged `Region2`, `ISP` physical map, `DarkEV2Box` and original raw-coordinate projection. Source blobs match accepted T022-C commit `824f36d9a7614acffa88977c4d128d0aa7afc85b`.
- Hard quadrant geometry, active dark EV `[0,2]`, active bright EV `[-0.5,0]`, active gamma `[0.8,1.25]`, inactive raw identity and exact pixel bypass. The raw EV map is `2*tanh(raw)`; its +2 physical endpoint corresponds to unbounded positive raw, exactly as T022-C. No new finite raw cap is introduced.
- Two deterministic starts: raw identity and accepted T022-C selected raw. Adam `lr=.05`, default betas `.9/.999`, epsilon `1e-8`, no weight decay/scheduler, exactly500 updates each. Seed7, TF32 disabled.
- Full native H400 W600 RGB MSE, float64 accumulation through the unchanged float32 renderer. Retain earliest minimum over steps0..500. Across starts choose lower MSE; identity wins exact ties. No metric-based optimizer changes or extra starts.
- Every result is labeled REFERENCE_ORACLE_ONLY; no state or target is passed into deployable TTT, source training or test-time model selection.

## Required structural and result evidence

The accepted selected-state rendering, physical grid and saved PSNR/SSIM must agree within1e-6. Retaining step0 requires oracle MSE no worse than accepted selected MSE within1e-10 for every image. All outputs, raw states, physical parameters and metrics must be finite. Bounds use1e-6 numerical tolerance. The pair allowlist limits image decoding to the exact200 validation low/normal paths.

Save both starts' loss histories, best/final raw states, best steps, winning state, physical grid, full float output, hashes and metrics. Report raw/selected/oracle means/medians; paired delta mean/median/p10/p90; winner and best/final-step histograms; active-only and all-coordinate EV/gamma saturation with1e-6 tolerance. Inactive coordinates are explicitly separated from active saturation denominators.

An independent script using only saved CSV/JSON and Python statistics recomputes summaries and verifies the numerical invariants without importing the optimizer. Full outputs remain in remote artifacts and F-disk backup; compact states/histories/metrics/receipts are reviewable in Git.

## Execution and environment

Release `20260914-075755-ttie-t025a-oracle`, job `20260914-075823-ttie-t025a-oracle`, physical A6000 GPU1. Torch2.4.0+cu121 passes an actual CUDA computation despite an NVML driver/library warning from nvidia-smi. No system driver changes were made.

Baseline focused tests:3 passed. Isolated oracle step-zero/projection/inactive-pixel test:1 passed. See local_checks and reuse receipts. Full100-pair run is the end-to-end check; no second scientific run is planned.

Interpretation is descriptive only. Any measured oracle gap demonstrates useful reachable states beyond the learned trajectory; modest remaining quality would implicate the current state family. MSE minimization may reduce SSIM on individual images. Stop after reporting; no promotion or new method change.
