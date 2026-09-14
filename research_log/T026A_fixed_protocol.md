# T026-A fixed active-gamma bound probe

One global validation-tuned change: active Region2 gamma lower0.8 to0.5. EV bounds, gamma upper1.25, inactive identity, renderer, nuisance gate, CLIP/prototypes, T014 energy, Adam0.03,40updates and earliest minimum predicted-energy selection stay frozen. No oracle artifact/metric/reference enters inference.

Use the100 frozen validation low images exactly once on A6000 physicalGPU1. Split SHA256 b88c8347005984b5523b117b52c0c068672fe172eb7c9aa5b60102d350e2d85b; frozen T014 energy c3d1eef9f20af163e1268823cf16d3fdaed315e7723fc94db3b1138ad1336521. Low-only executable accepts no normal/reference root; input image opens are allowlisted and logged. Task-specific shared/t026a/normal remains absent until all100 outputs/decisions/trajectories have been persisted and frozen. Other-task reference data stays outside this input allowlist. Do not claim global server reference absence.

Post-freeze evaluation uses accepted RGB float full-frame PSNR and Gaussian11/sigma1.5/population/full-map-reflect SSIM, with independent implementation agreement within1e-11. Compare only accepted T022-C. Materially positive requires paired meanPSNR>=+0.50dB and meanSSIM>=0.3282314776612914 (1e-12 numerical tolerance). No other candidate, retraining, additional run or official-test access.

Compare all100 gates and boxes, require only active gamma lower differs, preserve selected and final physical-state saturation separately for active/inactive coordinates (tolerance1e-6). Report per-image metrics/deltas, mean/median/p10/p90, step histogram, runtime mean/median/p95, image improve/worsen counts, and independent aggregation. Exact0.5 maps to unbounded negative raw in the existing renderer; no finite raw cap is introduced.

Focused tests: accepted baseline3passed; variant3passed covering exact bound/finite renderer endpoint, trajectory/runner AST equivalence after documented wiring/metadata normalization, and100 synthetic normal-file mutation/withholding with identical output/decision/trajectory hashes. Real100-image inference is not repeated for these tests. No T025 per-image state or reference-derived statistic is loaded.
