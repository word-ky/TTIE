# T065-A fixed linear trajectory-quality head

Authorization e9e605bb05238c159dfa5d9ff45112228de7dad5; exact OPEN task in authorization.md. Exposed-cohort transfer only.

Reuse map: T064-B supplies frozen-state/low PNG loading, read-scope receipts, CommonRegion2 rendering, T063-C base selector, source/state hashes, post-freeze evaluator and independent metric audit. New code is limited to fixed11 features, standardization, one ridge solve and predicted-margin argmax.

Fixed arithmetic: float32 accepted GPU renderer and weighted objective; float64 GPU image/state statistics, float64 CPU objective progress, normalization and linear solve. Luminance=.299R+.587G+.114B. Spatial gradient is the mean of concatenated absolute valid horizontal and vertical forward differences (no padding); ratios use floor1e-8. Both std operations are population. State norms use all12 raw coordinates. p/q are unclipped exact stated fractions. Feature order is core.FEATURES.

Training: all100 original development images x28states, each sample equal weight. Targets use accepted Git-bound T062-B PSNR(state)-PSNR(T026). Independently recompute all development state PSNR/SSIM against normals; T026/T036 development anchors reuse accepted bound evidence. Standardize every11feature with population std floor1e-8. Solve (A^T A+diag(0,.001,...,.001)) beta=A^T d in float64; intercept unregularized. This is summed squared residual ridge, no mean-loss rescaling. Exactly one model, no search. Independent fit uses separate NumPy feature computation, einsum normal equations and SciPy positive-definite solver.

Freeze training table, normalization/model and source/state bindings before any transfer reference-quality read. Transfer reads100low PNGs+100frozen traces only; evaluate model on0..base, largest predicted margin and earliest exact tie. Freeze all100choices/output hashes before reference metadata/metrics. Report five unchanged gates, histogram/changed choices and post-freeze tail16/86. No optimizer rerun, feature revision, retraining, held-out/new cohort access.

Validation: baseline10passed10.88s. Initial Windows NumPy/torch numeric runtime aborted at matrix multiplication; MKL_THREADING_LAYER=SEQUENTIAL with single-thread BLAS resolved it without code/model changes. New3tests passed8.21s, expanded4including independent feature/fit equivalence passed8.46s. Final affected tests and single GPU audit follow.
