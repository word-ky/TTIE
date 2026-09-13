# T021-A progress

- 2026-09-13T16:05:49.549804+00:00: E accepted and T021-A opened at main dfad6022. PR14 accepted merge cebecffb; exact T014 Stage-B artifact origins copied. Main qualification has200 primary rows plus40 report-only offset rows. Multiple baseline definitions exist and no unique H0 binding is stated in T021-A. Clarification requested; do not calculate real SSIM or bootstrap until baseline/pool resolved. Recovering/hash-verifying all frozen output files without loading clean references.

- 2026-09-13T16:08:44.292683+00:00: All480 frozen output/decision hashes verified,3,924,677,163 bytes; no tensors/clean refs loaded. SSIM/full-map cross-check and cluster-bootstrap tests2 passed1.01s. Awaiting explicit baseline and primary pool; no real SSIM/CI calculated. Prepare partial PR and research mailbox clarification.

- 2026-09-13T17:20Z: Research clarification d7dc7ec8 binds matched value-only H0 versus Sobolev H1, 200 primary rows excluding offset. Added evaluation-only artifact replay and independent separable-kernel verifier; accepted metric/bootstrap unchanged. GPU preference recorded; this fixed SciPy metric remains CPU. Remote skimage missing; no installation required because independent verifier uses explicit SciPy convolution and was locally cross-checked.
