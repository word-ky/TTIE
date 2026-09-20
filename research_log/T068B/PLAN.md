# T068-B exact development objective-motion knee

Reuse internal accepted T068A/T067B/T066A artifacts, source 3f0f2d5cbca0ab13efbd7ddd1bdf5570c74cbd38 and evidence 5866895ee68c68b59d1ceb4e38140698e167fbe3. Baseline last completed: 10 tests passed local12.01s, remote1.83s; GPU independent2041 renders and CPU2800 states PASS. Same repo code, no new external licensing/dependencies.

Reuse map: T066A fixed inputs/model/state/renderer and independent development quality routines; T067B exact interpolation and endpoint candidate table; T063B unchanged summarizer. New code only knee formula, float64 RMS motion, curve/freeze and verifier orchestration. No training/optimizer run.

Increment1 focused core tests: singleton, loss/motion-degenerate, cumulative motion, clipping and larger exact tie. Increment2 run/verifier plus affected regressions, then source commit/push and full real development run. Primary RMS on GPU float64 from exact frozen float32 images promoted before subtraction. Independent verifier re-renders all interval states on GPU then recomputes RMS and curves from those tensors using NumPy float64 CPU. Require exact choice agreement and numerical curve agreement atol1e-12. Ranking uses actual primary float64 and exact larger-step tie, not epsilon ties.

Complete interval curves stored for all100 images, d_FS=0. For degenerate denominator, mathematically undefined u/v/a arrays are null entries, with reason and selected endpoint; d/s remain actual values. This bookkeeping does not change the specified early endpoint cases. Inputs finite and bounded via immutable accepted artifacts. Curves/selected state+render/model/rule hashes frozen before reading or hashing quality. Model and lambda=.875 unchanged. T067B frozen candidate table is the endpoint equivalence donor; no cap is applied.

Offline primary uses accepted development quality, independent CPU recomputes all28 states; report five gates and paired control differences without tuning. Source binding excludes evaluation quality; no transfer/fresh/final files or references. Classify changed>=1 and allgates as CANDIDATE_FROZEN, otherwise NO_GAIN. Stop/report exact rule.
