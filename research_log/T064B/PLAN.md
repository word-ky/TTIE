# T064-B fixed antithetic sensitivity audit

Authorization: 542e36375e58cdba2e71149ec6a751e8db47bc58. Execute only the exact OPEN task in authorization.md.

Reuse CommonRegion2 and frozen T062-A development/T063-D transfer states, no optimizer. Baseline tests: 7 passed; new statistic/guard tests: 3 passed. GPU 1 A6000 for all renderer calls.

Canonical perturbation: CPU torch.Generator manual_seed(64064), int64 randint(0,2) per full tensor shape, signs=2*r-1, cast to input float32 GPU. Delta=1/255; clamp both perturbed inputs. GPU float32 mean absolute differences; Python binary64 ratio with denominator floor1e-8. Same perturbation tensor for same shape, independent of image ID. Independent verifier regenerates pairs and hashes all 28 states.

Fixed normalized-progress rho=0.9857470621423519. Search largest accepted sensitivity step within base prefix; fallback identity. Candidate grid contains exactly unique finite development sensitivities in each base prefix, no sentinels. Eligibility requires all five gates; maximize absolute development mean PSNR, tie smaller tau. Freeze candidate table, rule, bindings and development choices before transfer. If none eligible, stop CALIBRATION_NEGATIVE and independently verify; no transfer reconstruction or quality reads.

For an eligible threshold, read only low PNGs and frozen traces during transfer selection. Freeze all 100 chosen outputs before reference metadata/quality access. Post-freeze report gates, histogram, changed choices and tail16/86 outcomes. Independent metric audit uses separable SSIM and dot-product MSE. No threshold retry or additional experiments after outcomes.
