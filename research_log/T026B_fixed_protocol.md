# T026-B fixed80-step budget probe

Use accepted T026-A, changing only max_steps40 to80. Same identity initialization, Adam0.03, gamma[0.5,1.25], EV dark[0,2]/bright[-.5,0], inactive identity, Region2 geometry, gate/CLIP/prototypes/T014 Sobolev energy, renderer and earliest minimum predicted-energy selector. The trajectory implementation is reused without edits.

Exactly100 native validation low images, once, on A6000 physicalGPU1. Split SHA256 b88c8347005984b5523b117b52c0c068672fe172eb7c9aa5b60102d350e2d85b. Runner only accepts low-root/split/assets/out; records100 allowed low opens and freezes all outputs/decisions/81-state trajectories before task-specific shared/t026b/normal deployment. Other-task references remain outside the input allowlist. No T025 oracle files are staged or read; no official test is accessed.

Post-freeze metrics use the accepted RGB float full-frame PSNR and Gaussian11/sigma1.5/population/full-map-reflect SSIM. Compare only against T026-A. Joint gate: mean paired PSNR gain>=0.50dB and meanSSIM>=0.373791825 with1e-12 tolerance. Exactly this rounded SSIM floor comes from the task; full baseline values remain in paired reports. No other budget, method change or follow-on experiment after the verdict.

Reuse all100 gate/box/asset equalities and frozen artifact hash checks, finite selected/final states and bounds, independent metric implementation and Python-statistics aggregation. Report raw/A/B mean+median, deltas mean/median/p10/p90,0..80 histogram, runtime mean/median/p95, selected/final active/inactive EV/gamma saturation, improve/worsen counts and worst regressions. Source diff and focused tests document only-budget equivalence.
