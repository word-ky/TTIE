# T028-A fixed reference-oracle audit

REFERENCE_ORACLE_ONLY. Research inbox3b7d473b/main50dcc759. No deployable change,
baseline execution, official-test filename/content access, metric-guided tuning
or subsequent experiment. Stop after the fixed audit and its report.

Reuse accepted T025-A optimizer (Adam .05, betas .9/.999, eps1e-8, weight_decay0,
500updates/start, step0 included, full-frame RGB MSE accumulatedfloat64 through
float32 renderer), wiring Gamma05Box from accepted T026-A. Preserve Region2,
frozen gate, darkEV[0,2], brightEV[-.5,0], gamma[.5,1.25], inactiveidentity.
Seven accepted renderer/bounds/metric Git blobs are unchanged. No learned energy
or CLIP calculation in oracle optimization. Never import oracle artifacts into
deployable code, model training, gate, selector or official-test paths.

Accepted T026-A source b2359721c89db732d17e03be273e0bdb71bb377a,
merge51f84a9d96880bca8e908c9b3cdff496d7277e39,
run20260914-113853-ttie-t026a-gamma05. Frozen split
b88c8347005984b5523b117b52c0c068672fe172eb7c9aa5b60102d350e2d85b,
100validation pairs and exact order. Use existing named low paths, no dataset
traversal or access to official-test filenames/images.

Preflight process accepts no reference-root argument. Before any reference
pixel, bind accepted freeze/config/metrics and per-image decision/output/
trajectory hashes, reconstruct all100 outputs and physical grids with exact
zero error, persist both identity and accepted-selected raw starts. Only those
100low images can be decoded. Task-specific reference deployment occurs after
the completed preflight timestamp. Existing other-task reference directories
are outside preflight access, not claimed globally absent.

Oracle process consumes bound starts and exactly100 named validation normals
inside the quarantined task area. Each of200starts once, Adam .05 x500updates;
save all501raw states and MSE values. Check finite outputs/raw/gradients and
physical bounds at every state. Pick smallest MSE across both histories, then
earliest step, then identity for any remaining exact tie. Record actual winner,
step, times and hashes. Freeze all100outputs/200histories before quality scoring.

After freeze: unchanged native full-frame RGB PSNR/Gaussian11sigma1.5SSIM.
Independent separable-convolution SSIM/Torch PSNR cross-check, saved history
argmin/bounds audit, compare accepted T026-A metrics, and independent Python
statistics/NumPy aggregation. Report means/medians/p10/p90 paired deltas,
winning starts/steps and active/inactive EV/gamma boundary hits (tolerance1e-6).

Interpretation fixed before outputs: substantial within-family headroom iff
paired mean PSNRgain>=2dB and median>=1dB; limited if mean<1dB; otherwise mixed.
This finite two-start local search demonstrates reachable states, not a
certified global optimum or deployable result. MSE optimum can reduce SSIM.

Baseline4tests passed24.75s; focused oracle test1passed17.85s (step0 retention,
projection, inactivepixelidentity, stored state-history alignment). Initial
sparse checkout lacked the older oracle source required by the baseline test;
including that tracked folder restored it. No method repair or tolerance
relaxation. All artifacts and reports remain in project-local research_log and
server REFERENCE_ORACLE_ONLY storage, with full F-drive backup.
