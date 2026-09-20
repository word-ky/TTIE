# T065-A — DONE / TRANSFER_NEGATIVE

Source commit: `326a685868b12990886fc910f36cdf4f29eaf2d3`. Branch: `codex/T065A-linear-quality-head`. Authorization: `e9e605bb05238c159dfa5d9ff45112228de7dad5`.

The single fixed linear quality head fails the immutable transfer worst-tail gate. It changes 1/100 development choices and 4/100 transfer choices relative to the frozen normalized-progress selector, but leaves both prior tail failures at steps21 and25. This closes this exact fixed11-feature ridge head; it is an exposed-cohort audit, not fresh qualification.

Transfer absolute mean PSNR/SSIM: 15.466282683512 / 0.397757988804. Training RMSE of the paired PSNR margin: 2.257578096782 dB over2,800 development state samples. All five development gates pass.

| Gate | Required | Development | Transfer | Transfer pass |
|---|---:|---:|---:|---|
| mean_delta_psnr | >=2.00 dB | 3.5651284617598207 | 3.85636781478698 | True |
| median_delta_psnr | >0 dB | 3.093839109589621 | 3.814406659510415 | True |
| regressions_t026 | <=29/100 | 3 | 12 | True |
| worst_delta_t026 | >=-5.614 dB | -4.074446413712419 | -10.364944679494553 | False |
| mean_delta_ssim | >=-0.001 | 0.014882602931350679 | 0.018282965250306236 | True |

Selected-step histogram (nonzero): {"13": 3, "18": 1, "19": 1, "21": 3, "24": 3, "25": 5, "26": 19, "27": 65}.

Changed transfer choices (index: base -> selected): 0: 25 -> 19, 1: 26 -> 13, 4: 26 -> 13, 47: 26 -> 13.

Post-freeze tail outcomes:

| Index | Step | Selected PSNR / SSIM | True margin vs T026 | Predicted margin |
|---|---:|---|---:|---:|
| 16 | 21 | 15.243922691784 / 0.730831917479 | -7.131115430332 | 4.339596252987 |
| 86 | 25 | 13.247833801770 / 0.607845717517 | -10.364944679495 | 5.145553307556 |

Implementation: reuse accepted T064-B data boundary/render/evaluation machinery and T063-C base checkpoint with rho=0.9857470621423519. New code computes exactly the authorized11features and one global ridge solve. All100development images and28states receive equal weight. Labels are accepted Git-bound development PSNR(state)-PSNR(T026); no transfer supervision. Everyfeature uses development mean/population std with floor1e-8. One float64 solve of summed squared residuals plus lambda=.001 times coefficient squared norm, intercept unregularized. No search/CV/model retry. Full normalization and coefficients are in `evidence/model.json`; complete features/targets in `evidence/training_table.json` and CSV.

Numerics: float32 GPU renderer and zero-reference objective; float64 GPU luminance/image/state features, float64 CPU progress/normalization/ridge solve. Y=.299R+.587G+.114B. Gradient magnitude is the mean of concatenated absolute valid H/W forward differences, no padding; ratio denominator floor1e-8. Image std is population; raw norms include all12coordinates. Predicted-margin argmax is restricted to0..base with earliest exact tie.

Model SHA256 `5a644c74b7b98a242e7c2945f2fa7b971a5d0e046ba3e66f4e5e7462ba7599ef`. Model/training/normalization/source/state manifest frozen 2026-09-20T10:13:13.669155+00:00; all100transfer choices/output hashes frozen 2026-09-20T10:14:44.946952+00:00; first transfer reference-quality read 2026-09-20T10:14:45.019689+00:00. Selector SHA256 `b138ec45ea3b689553faec6720a64011740fd962b52666fe1a081ab4cf294bb4`; transfer freeze SHA256 `c0f5c7fb1fb8ab0cf3620246d17539f05deef4705dbf0bc47ad855dfa3217092`. Recorded feature/selection read sets contain only each cohort's100low PNGs and100frozen traces. Transfer feature table contains target-free features/predictions only, with metadata/hash receipts; no reference outcome fields. No optimizer rerun, new cohort, official LOL-v2 test or cross-dataset read.

Independent verification PASS: re-render all2,800states per cohort; recompute features with separate NumPy operations and solve normal equations using independent einsum/SciPy arithmetic; reproduce model, predictions, exact choices, output hashes, read ordering, metrics and final classification. Maximum feature errors development/transfer 7.105427357601002e-15 / 6.217248937900877e-15; coefficient error 1.0120881910324897e-10; development/transfer metric error 1.0871303857129533e-12 / 8.846257060213247e-13. Development state metrics are recomputed against originals; accepted Git-bound T026/T036 development anchors are reused. Transfer controls are independently scored and output hashes checked.

Validation: 10 baseline passed10.88s;3 unit passed8.21s;4 cross-implementation passed8.46s;11 affected passed11.92s; remote11passed1.53s. Initial local NumPy/torch numeric runtime abort was resolved by single-thread sequential MKL, with no method change. No remote failure or scientific retry. Source was pushed before execution. Run `20260920-181135-ttie-t065a-linear`, physical A6000 GPU1, exit0, primary pipeline 185.499915s. Model/render operations use GPU; the12x12ridge solve and independent numerical metrics use CPU.

Commands: `python -B -m pytest research_log/T065A/test_core.py research_log/T063C/test_core.py research_log/T063A/test_core.py -q`; `python -B -m research_log.T065A.run --out /media/wenchang/F/wjq/TTIE/runs/T065A-linear-quality-head`; `python -B -m research_log.T065A.verify --out /media/wenchang/F/wjq/TTIE/runs/T065A-linear-quality-head`. Environment: CUDA_VISIBLE_DEVICES=1, TF32off, CUBLAS_WORKSPACE_CONFIG=:4096:8, OMP/MKL/OPENBLAS threads1, MKL_THREADING_LAYER=SEQUENTIAL.

Artifacts: `research_log/T065A/evidence/` contains model, development/transfer features, training table, frozen receipts, per-image metrics, tail outcomes, config, verification and runlog. Raw `/media/wenchang/F/wjq/TTIE/shared/t065a/T065A_raw.tar`, SHA256 `5f7f1732eb5678258f9022aa1207529473ff6c53ca76315c03c5b5a78361782a`, 293632000 bytes. Recovery `/home/wenchang/asdasdsad/wjq/TTIE/shared/t065a/T065A_recovery.tar.gz`, also on F: and fetched locally, SHA256 `df7ddf02c7a0116a352838cad33d33cc6f930a963867972136961d46edc71f25`, 2243400 bytes. Home/F/local recovery hashes match.

Recommendation: accept TRANSFER_NEGATIVE and close this exact fixed linear head. Its slight selection changes do not discriminate the unsafe late checkpoints. Await a new bounded task from the research lead; no retraining, new features, alternate model or held-out access in this cycle.
