# T074-C near-black target preregistration (sealed before any method runs on these targets)

2026-09-26 +08. Execution delegated to the assistant by the user; defaults below may be overridden by the user/research lead only before the affected rows run.

## Candidate measurement (sealed criterion `research_log/T074A/darkness_criterion.md`)

Staged with `research_log/T074B/stage_target.py` (LF SHA256 `7f6ff8bc901eee6cbce987535f9197a6342442788883e6dbe1c7ae9e23c9b902`; the only change from the reviewed version removes the stager's own per-frame basename-equality check, because SDSD LQ/GT frame numbers differ by a constant per-video offset and the official SNR loader pairs positionally). Low images only; GT byte-hashed, never decoded.

| Candidate | n | clusters | geometry | median Y | frac Y≤0.15 | result |
|---|---|---|---|---|---|---|
| SDSD-indoor (6 test videos × first 30) | 180 | 6 | 960×512 | 0.0890 | 1.000 | **PASS** |
| SMID (49 seqs × first 30) | 1470 | 49 | 960×512 | 0.0939 | 0.653 | FAIL |
| LSRW Eval (Huawei+Nikon) | 50 | 50 | native 960×640/720 | 0.0781 | 0.720 | FAIL |
| LOL-v2 Real Train Low (anchor only) | 689 | — | native | 0.0560 | 0.955 | — |
| SID-sRGB | — | — | 960×512 | pending download | | pending |
| SDSD-outdoor | — | — | — | no loader-compatible source | | not measured |

pHash overlap vs LOL-v2 Real Train (Low+Normal): 0 flagged for all three measured sets. Channel order (loader-delivered RGB) confirmed by visual inspection of 8 brightened LOW images per set (SDSD-indoor, SMID): loader order natural, R/B-swapped unnatural.

## Targets

Main-table targets = every candidate that passes the sealed criterion: **SDSD-indoor** now; **SID-sRGB** is added if and only if it passes when measured (decided by the same code, before any method runs on it). Failing sets are reported as measured and excluded.

## Rows (per target)

Non-tuned rows, all frozen and verified before any target GT is decoded: RetinexFormer, SNR-Aware (LOL-v2 Real source checkpoints), PromptIR (five-task `epoch=80.ckpt`), PromptIR+DCTTA (same base, domain-level over the target's lows, order sealed by `--order-only` first), MR. Illuminate, QuadPrior (official 512-short-side + bilinear map-back to low geometry), Ours-Step0 and Ours-TTT (frozen T070-A + sealed no-active abstention). Then the separate "Ours-TTT (target-tuned)" row per `research_log/T074A/ours_target_tuning_decision.md`.

## Metrics / statistics (fixed now)

- RGB PSNR / RGB-SSIM with the T072-L/T071-B implementation; every row's output clipped to [0,1] at metric time; GT converted through the same official SNR loader (960×512, /255, channel swap) as the lows; assert low/GT geometry equality per pair.
- Mean and median over images; paired deltas `Ours-TTT − comparator`, `adapted − base`; win fraction (strict >).
- Cluster bootstrap by video/scene, seed 20260922, 10,000 resamples, image-weighted ratio statistic. SDSD-indoor has only 6 clusters: its CIs are reported but flagged as low-power.
- Ours no-active abstention counts reported per target.
