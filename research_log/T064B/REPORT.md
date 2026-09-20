# T064-B — DONE / TRANSFER_NEGATIVE

Source commit: `903c45727ad0d8d1d64e86bf4f2b73665446dc03`. Branch: `codex/T064B-antithetic-guard`. Authorization: `542e36375e58cdba2e71149ec6a751e8db47bc58`.

The fixed antithetic-sensitivity guard failed to transfer. Calibration chose tau=16.09636904055864 from exactly 2,663 unique finite development prefix sensitivities. It changed 0/100 transfer choices, so the prior worst-tail failure remains. This is an exposed-cohort audit, not fresh qualification. Close this exact guard without changing the seed, delta, formula, threshold or cohort.

Development changes vs base: 0/100; absolute mean PSNR 14.799572435166 dB. All five development gates pass. Transfer absolute mean PSNR/SSIM: 15.471845204573 / 0.397896864536.

| Gate | Required | Development | Transfer | Transfer pass |
|---|---:|---:|---:|---|
| mean_delta_psnr | >=2.00 dB | 3.5695322981528443 | 3.8619303358477532 | True |
| median_delta_psnr | >0 dB | 3.093839109589621 | 3.814406659510415 | True |
| regressions_t026 | <=29/100 | 3 | 12 | True |
| worst_delta_t026 | >=-5.614 dB | -4.074446413712419 | -10.364944679494553 | False |
| mean_delta_ssim | >=-0.001 | 0.015011775217397242 | 0.01842184098158023 | True |

Selected-step histogram (nonzero): {"18": 1, "21": 3, "24": 3, "25": 6, "26": 22, "27": 65}.

Post-freeze tail outcomes:

| Index | Step (base unchanged) | Selected PSNR / SSIM | Delta PSNR vs T026 | Sensitivity at choice |
|---|---:|---|---:|---:|
| 16 | 21 | 15.243922691784 / 0.730831917479 | -7.131115430332 | 3.913695024084 |
| 86 | 25 | 13.247833801770 / 0.607845717517 | -10.364944679495 | 5.316773535034 |

Implementation and boundary: reuse accepted CommonRegion2 renderer, frozen T062-A development and T063-D transfer states, zero optimizer calls. Fixed rho=0.9857470621423519, seed64064, delta1/255; CPU seeded Rademacher generation per tensor shape, float32 GPU renders/means and binary64 ratio. Largest permitted sensitivity step in the normalized-progress prefix; identity fallback. All five gates required for candidate eligibility; maximize absolute development mean PSNR and tie smaller tau. No sentinel candidates.

Selector frozen 2026-09-20T08:37:46.340670+00:00; all 100 transfer choices/output hashes frozen 2026-09-20T08:38:25.182246+00:00; first transfer reference-quality read 2026-09-20T08:38:25.222798+00:00. Selector SHA256 `6de08549adcdada7e14061cf4fcfee559cf42482ec22e36b8321ed336cbf152a`; transfer freeze SHA256 `af968013c656f6c2ee281e4b2738c52149abd0c37d1ab2aa8c23a85dc20c1925`. The recorded selector read set contains only 100 low PNGs and 100 frozen T063 traces. No transfer normal, baseline outcomes or oracle inputs in selection. No new cohort, official test or cross-dataset reads.

Validation: local 7 baseline tests passed (12.27s), 3 new tests passed (11.18s), final 10 affected tests passed (9.64s); remote 10 passed (1.52s). Independent verifier PASS: regenerate all antithetic inputs/pairs and 2,800 states per cohort; reproduce all statistics, exact candidate grid, tie-break, choices, hashes, read ordering, metrics and classification. Maximum development metric error 1.0871303857129533e-12, candidate error 2.7355895326763857e-13, transfer metric error 8.846257060213247e-13.

Run `20260920-163659-ttie-t064b-antithetic`, exit0, physical A6000 GPU1. Primary calibration/transfer/evaluation 81.829310s; no optimizer reruns. Commands: `python -B -m pytest research_log/T064B/test_core.py research_log/T063C/test_core.py research_log/T063A/test_core.py -q`; `python -B -m research_log.T064B.run --out /media/wenchang/F/wjq/TTIE/runs/T064B-antithetic-guard`; `python -B -m research_log.T064B.verify --out /media/wenchang/F/wjq/TTIE/runs/T064B-antithetic-guard`.

Failures/deviations: two initial remote preflight collection failures due to omitted accepted T063-A test dependencies (reconstruct/analyze/common) in the release package. Both failed before scientific execution; packaging-only commits fixed them. Final scientific source was committed/pushed before the sole scientific run. No scientific design deviation or post-outcome retuning.

Artifacts: `research_log/T064B/evidence/` contains full development sensitivities/thresholds, selector manifest, transfer freeze, per-image metrics, tail outcomes, verification, config and run log. Raw server archive `/media/wenchang/F/wjq/TTIE/shared/t064b/T064B_raw.tar`, SHA256 `8c6b27834d25a22418341b8488e51ae46beeec5d391d148c86cb56720c8bb6e7`, 292536320 bytes. Recovery `/home/wenchang/asdasdsad/wjq/TTIE/shared/t064b/T064B_recovery.tar.gz`, also on F: and fetched locally, SHA256 `76700478a68bc528aa5df489f8b2adb0e5bcb271f39f89c7c1e625035bee354d`, 1888978 bytes. Home/F/local recovery hashes match.

Recommendation: accept TRANSFER_NEGATIVE and close this exact antithetic guard. Await research-lead review/new bounded authorization; do not infer an oracle step cap or try another magnitude/seed.
