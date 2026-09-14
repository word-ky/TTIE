# T025-A: frozen Region2 reference-oracle reachability

**REFERENCE_ORACLE_ONLY — oracle ceiling measured.** Exactly100 frozen validation pairs completed both fixed starts (200 starts,100000 Adam updates) on NVIDIA RTX A6000. Every selected start reproduced the accepted T022-C result within tolerance, every oracle MSE was non-worse, and independent aggregation passed. No official-test image was opened; no deployable module, model weight, gate or action bound changed.

This is a non-deployable reference-assisted diagnostic. The fixed two-start local search demonstrates reachable quality in the existing action family; it does not certify the global optimum or an absolute mathematical upper bound. No oracle state or reference-derived quantity is fed into deployable TTT.

## Measured validation quality

| Output | Mean PSNR dB | Median PSNR dB | Mean RGB-SSIM | Median RGB-SSIM |
|---|---:|---:|---:|---:|
| Raw input | 8.109722672 | 7.600161541 | 0.160022843 | 0.138977265 |
| Accepted T022-C | 10.229554025 | 9.650373830 | 0.328231478 | 0.300251538 |
| REFERENCE_ORACLE_ONLY | 13.545967045 | 12.351664556 | 0.384052485 | 0.389328584 |

| Oracle minus T022-C | Mean | Median | p10 | p90 |
|---|---:|---:|---:|---:|
| psnr | +3.316413020 | +2.340446425 | +1.034132063 | +6.740197890 |
| ssim | +0.055821007 | +0.054811715 | -0.006111037 | +0.114779785 |

PSNR improves by more than1e-6 on 100/100 images. SSIM decreases on 18/100; SSIM was scored after optimization and never enters the MSE objective or state selection.

## Fixed search and state provenance

The original deterministic split SHA256 is `b88c8347005984b5523b117b52c0c068672fe172eb7c9aa5b60102d350e2d85b`. Reused T022-C run: `20260914-042925-ttie-t022c-ev2`. Its decision/output/trajectory hashes are checked against the existing freeze receipt per image. Each gate remains frozen; the reconstructed action box must equal the saved box. Native RGB600x400, unchanged hard Region2 rendering, active dark EV0..2, bright EV-.5..0, gamma.8..1.25, inactive identity.

Seed7; exactly500 Adam updates per identity/selected start, lr0.05, default betas/epsilon, no scheduler. Float64 full-frame MSE accumulation through the accepted float32 renderer. Retain earliest minimum at steps0..500 and lower-MSE start; identity resolves exact inter-start ties. No additional starts, schedule changes or bounds tuning.

## Start and step statistics

Start wins: `{"T022C_selected": 82, "identity": 18}`.

- best_step: `{"138": 1, "180": 1, "183": 2, "184": 1, "185": 1, "200": 1, "204": 1, "214": 1, "227": 1, "232": 1, "235": 1, "262": 1, "282": 1, "283": 1, "319": 1, "361": 1, "491": 1, "500": 69, "7": 13}`
- identity_best_step: `{"185": 1, "186": 1, "189": 1, "206": 1, "207": 1, "210": 1, "214": 1, "217": 1, "243": 1, "250": 1, "255": 1, "264": 1, "282": 1, "309": 1, "319": 1, "352": 1, "491": 1, "496": 1, "500": 69, "7": 13}`
- selected_best_step: `{"138": 1, "14": 13, "162": 1, "180": 1, "183": 2, "184": 1, "200": 1, "204": 1, "205": 1, "227": 1, "232": 1, "235": 1, "240": 1, "262": 1, "283": 1, "293": 1, "361": 1, "382": 1, "500": 69}`
- final_step_all_starts: `{"500": 200}`

A winning best state at step500 means the fixed search was still finding its lowest observed MSE at the budget endpoint; it is not proof of convergence. This task does not add steps or restarts in response.

## Bound saturation

Tolerance is1e-6 in physical coordinates. Active-only denominators exclude the trivially fixed identity coordinates; all-coordinate denominators are also reported in the JSON. The +2 EV endpoint uses the unchanged2*tanh(raw) map and can be approached without being counted as saturated at this tight tolerance.

Active regions: 392; inactive: 8; total400.

| Parameter bound (active only) | Count | Denominator | Fraction |
|---|---:|---:|---:|
| ev_lower_active_count | 0 | 392 | 0.000000 |
| ev_upper_active_count | 86 | 392 | 0.219388 |
| ev_either_active_count | 86 | 392 | 0.219388 |
| gamma_lower_active_count | 388 | 392 | 0.989796 |
| gamma_upper_active_count | 0 | 392 | 0.000000 |
| gamma_either_active_count | 388 | 392 | 0.989796 |

## Checks, execution and interpretation

Maximum selected-output reproduction error: 0 (tolerance1e-6); physical-grid and PSNR/SSIM reproduction checks also pass. All100 oracle MSE values satisfy non-worse tolerance1e-10. All200 starts execute500 updates and all outputs/metrics/states are finite. Independent CSV/JSON aggregation passes without importing optimization code.

Mean per-image runtime11.852s; median11.882s. Job `20260914-075823-ttie-t025a-oracle`, release `20260914-075755-ttie-t025a-oracle`, source `04e436dbefad747a562fd68faacecaf5d90006c9`. Actual CUDA computation and full run pass despite the server NVML monitoring warning. Baseline3tests and focused oracle1test passed before launch.

The measured mean gap is +3.3164dB / +0.055821 RGB-SSIM. This quantifies useful states found outside the accepted learned trajectory under the same gate/action family. The absolute oracle quality, paired distribution, boundary use and endpoint-step counts must also inform whether the state family remains limiting. No binary promotion claim or SOTA claim is made; the research lead chooses the next task.

## Descriptive frozen-gate groups

These groups use the original low-only gate labels, not reference-derived relabeling. They describe heterogeneity and do not change any image or method.

| Frozen active gate types | Images | Mean delta PSNR | Mean delta SSIM | Oracle mean PSNR | SSIM worse |
|---|---:|---:|---:|---:|---:|
| dark_only | 73 | +3.878814 | +0.074440 | 14.281982 | 4 |
| mixed | 14 | +1.908767 | +0.004953 | 12.573227 | 11 |
| bright_only | 13 | +1.674240 | +0.006051 | 10.460527 | 3 |

All 86 active bright-winner EV coordinates remain at their allowed upper bound 0. The 306 dark-winner EV coordinates have median 1.997626245, near +2 but none meets the tight1e-6 upper-saturation tolerance. Therefore the aggregate EV-upper count86 refers to bright-winner zero bounds, not dark +2 endpoints. Gamma lower saturation is388/392 active coordinates.

This is evidence for both reachable improvement within the frozen family and strong boundary use; it does not by itself distinguish a certified action-family optimum from finite-search limitations.

## Evidence locations

- `research_log/T025A_fixed_protocol.md`, `T025A_reuse.json`, `T025A_local_checks.json`, launch/deployment receipts: fixed settings and exact reuse.
- `research_log/remote_runs/20260914-075823-ttie-t025a-oracle/artifacts/REFERENCE_ORACLE_ONLY/`: config, completion, provenance, per_image.csv/json, independent_summary.json, independent_aggregation.json,100 per-image oracle_states.json files with both loss histories and best/final states.
- Original remote run contains100 float oracle_output.pt files; their byte hashes are in provenance. Full run plus source release is backed up on F disk; compact evidence is fetched into this project. See T025A_artifact_receipt.json for exact archive hashes.

No changes to T022-C/T023-A, no baseline inference, no official test, and no further optimizer trial. Stop for research-lead review.

oracle ceiling measured
