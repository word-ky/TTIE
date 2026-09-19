# T060-B common-gain source direction audit

Classification: **T059-E is a common-gain direction candidate for one later finite-step test**. This is source-only first-order evidence; no finite-step quality or real-domain safety claim.

Authorization ed214057e12994d6bb4225fa909c75d792b4498a. Tested source 73b86dc14b0a4d80d0dad895407b132f30b83976; source parent2fb48219b52b07f9ab42ea312190dcb401b10b5a. Branch codex/T060B-common-direction. Sole run 20260919-114102-ttie-t060b-common, physical A6000 GPU1. Six focused tests passed3.23s; prior A baseline4passed3.29s and gain-only equivalence2passed1.68s. Full prediction, separate diagnostic and independent verification exit0.

## Fixed comparison

The exact80 source-outer state0 anchors retain their order. A includes only any(frozen gate.active):60/80, selection frozen 2026-09-19T03:34:59.196694+00:00, SHA256 494fdafcf0db462d7d8157175bd50a879217f6e5f0c973c9af833b50ac726297. No reference norms, scores or harmful labels select A. All60 are nondegenerate under the unchanged three-norm >1e-12 rule. Smallest reference norm 2.740120211381937e-12; this threshold was not changed.

| Metric, same60 anchors | T059-E | Original T014 |
|---|---:|---:|
| Positive dot fraction | 0.9666666666666667 | 0.8833333333333333 |
| Wrong-sign count | 2 | 7 |
| Cosine mean | 0.7774807272689549 | 0.6290119788169972 |
| Cosine median | 0.906366178533033 | 0.80090100474288 |
| Cosine p10 | 0.42592417291029977 | -0.060745921633695375 |
| Cosine p90 | 0.9997704433416597 | 0.9887045827732086 |

Exact deltas: {"positive_fraction_E_minus_014": 0.08333333333333337, "median_cosine_E_minus_014": 0.10546517379015308, "wrong_sign_014_minus_E": 5}. All preregistered gates pass: positive fraction>=.90, median>=.60, neither direction statistic worse, median advantage>=.05 and wrong-sign reduction>=2 (both material alternatives pass).

## Provenance and information boundary

Unchanged T036 CommonRegion2/Region2/ISP, literal RGB-shared gain with physical exp(ln2*tanh(raw_gain)), hard2x2 mask. Only gain raw[2:3] is differentiable; EV/gamma raw/grid are fixed. Every identity y0/grid matches the pinned U/A state hashes exactly. One fresh CLIP28-D feature/gain Jacobian feeds both heads. Each uses its own original checkpoint normalization and accepted load_energy semantics. Both heads remain immutable; all244 bound source files and all input hashes pass before/after.

E checkpoint e15e91c4e9be401bcac6d3de039ec2d1ae7141fb6cec40c388d22549652d43d0; T014 checkpoint c3d1eef9f20af163e1268823cf16d3fdaed315e7723fc94db3b1138ad1336521. T036 Git identities/SHAs in provenance.json, original cohort SHA f3fabdec3e9397a133347094a5cd619153f50c9b5cd03e4348980b5a37534fd4.

Target-free start 2026-09-19T03:41:10.347609+00:00; all60 fields persisted/fsynced/hash frozen 2026-09-19T03:41:33.046537+00:00; first source-clean read 2026-09-19T03:41:39.024454+00:00. Inference filesystem firewall, clean/reference/reference-gradient/target-domain/LOL-v2/official-test/leakage counters all0. Separate diagnostic opens16 source-training clean images; independent verifier reopens the same16, always afterfreeze. No optimizer step, finite-step output comparison, PSNR/SSIM, state selection, training or threshold tuning.

Independent verifier checks complete80 selection, ordered60, exact state/gate/features/hashes, explicit separately normalized SiLU MLP derivatives, NumPy J-transpose-q, analytic common-gain derivative with hard quadrant sums and clamp, scalar norms/dots/cosines/quantiles/deltas/classification. Maximum analytic reference absolute error 2.8805502864936994e-08; relative error 1.113312537761644e-06. Numerical tolerances were committed before the experiment.

## Failures, artifacts and next step

No scientific or execution failure, repair or rerun in the sole experiment. Preparation: an initially guessed test filename was absent (corrected to directory collection); Windows command payload exceeded argument length (switched existing SCP helper); intermittent SSH/SCP timeouts recovered before execution. Existing NVML warning did not prevent GPU execution. D free space recovered; local synchronization resumed.

evidence/ contains the full60-row alignment table, per-head norm distributions, compact x/J/q/g fields, source reference gradients, head normalization/state hashes, timestamps/manifests, independent checks and exact run command/log. Large low/base/prediction bundles are in the home/F raw archive. Result SHA256 bf4b42541919f17633b8cc372cd5e55ec342c560fed160b973f100ae084396fc.

Recommend research-lead review and at most authorization of the later finite-step common-gain test. Do not infer enhancement gains, tail safety or deployability. No real-domain or official-test run is authorized by this result; no self-merge.
