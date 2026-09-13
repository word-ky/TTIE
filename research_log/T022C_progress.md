# T022-C active

2026-09-13T20:25:03.230932+00:00

T022-B accepted merge5f042bfd1c7757089ab9a22014eeb857a88b1a24; current task main889fc33191268c787c1db9efbb26e3467d9e2ab1. Worktree .autodl/T022C-work, branch codex/T022C-ev-range.

Exactly one variant: active dark-winner Region2 EV upper0.5->2.0. Bright[-.5,0], inactiveidentity, gamma[.8,1.25], frozenhead/gate, Adam.03,40updates,min-energy selection unchanged. Same100validation split b88c8347005984b5523b117b52c0c068672fe172eb7c9aa5b60102d350e2d85b. No officialtest/sweep/retraining. All100 outputs frozen before new evaluation reference directory is populated. Low-only inputdecoder audit and synthetic100-target mutation/withholding test. Clone saved output views to avoid observed11.8GB backing-storage overhead without changing tensor values.

Exact endpoint preflight: float32raw10 maps exactlyto2 via unchanged2*tanh; existingActionBox.raw(2) is+inf, so onlythat raw upper clamp is nonbinding, physicalrenderer stillcaps2. No rendererchange/1.999substitution. Verify projectionandfinitegradient onGPU beforeexperiment.

Proposed reuse: explicitnew DarkEV2Box subclass; copy unchangedtrajectory body with onlyActionBox binding different; new benchmark runner importedvariant andmetadata, nohistoricalfileedits. Endpoint/box/trajectoryidentity and targetisolationtests first. Materiallypositive iff meanPSNRdelta>=.50 andmeanSSIM>=acceptedA. Otherwise negative/insufficient. Structurallyblockedif exactendpoint impossible (currentCPUtest succeeds).

- 2026-09-13T20:28:20.026171+00:00: Six historical baseline tests pass12.87s; three new endpoint/AST/100-target counterfactual tests pass11.30s. Unchanged acceptedtrajectory AST, historicalfilesbyteequal, onlydarkupper2 algorithm change. Targetcounterfactual uses syntheticfixtures, never reruns actualvalidationimages. Preparing GPUpreflight.

- 2026-09-13T20:29:30.338717+00:00: Source824f36d9a7614acffa88977c4d128d0aa7afc85b frozen. CPU+A6000 endpoint/projection, identicaltrajectory AST, and100synthetictarget counterfactual hash tests all pass remotely. Existing NVMLwarning does not affect CUDA. Generated unifieddiff context blanklines trigger genericgit whitespace warnings; source-code whitespace check passes. Formal100imageGPUrun launching once.

- 2026-09-13T20:30:31.404544+00:00: Formal run20260914-042925-ttie-t022c-ev2 started. Source files intentionally retain CRLF; ordinary diff --check flags carriage returns, while the established cr-at-eol source-code check passes. No source bytes changed. All100 real low images run once; synthetic isolation tests separate.
