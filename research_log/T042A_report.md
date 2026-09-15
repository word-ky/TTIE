# T042-A — DONE: late real legacy-state effect supported

**REFERENCE_GRADIENT_DIAGNOSTIC_ONLY.** On the same100 T036 real images/gates at fixed common gain1.75, substituting each trajectory's step10 legacy EV+gamma raises total positive-dot from **37% to94% (+57pp)** and median cosine from **-0.185847992683 to0.685276543900 (+0.871124536583)**. Both exceed the fixed +20pp/+0.25 criterion. The single verdict is **late real legacy-state effect supported**. No threshold, state or baseline was fitted after observing references.

This within-image substitution supports late real legacy/feature-state extrapolation as a major contributor and weakens a pure content-only account of T041. It does not establish a deployable checkpoint/stopping rule: these are isolated gradient probes with gain fixed1.75, not original step10 outputs, new optimization trajectories or a PSNR/SSIM comparison. No permission to deploy step10 is inferred.

## Provenance and fixed substitution

Executed/tested source **951e06671a74c3c2b91bd92646182029c723b7d0**, branch `codex/T042A-step10-legacy`, PR [#67](https://github.com/word-ky/TTIE/pull/67), base80078e23 after accepted T041 merge3bb7aa0a8561a03df573fde5f27bdbf04b420d8a. Evidence/main-mailbox SHAs are in final delivery. Reuse accepted T041 low-only gradient/evaluation loops, T036 model loading, T038 gate/alignment grouping, T039 probe_raw and T029 scalar conventions. No deployable source module changes.

Exact T036 cohort SHA **279c74b335999d6631d436c79ea80e9e2cccfc4b97cb7c0b992829d04cfaf40b**, accepted original freeze **46e667ded785e4f3f8ba341d40d00a7e02ca652e2778fcc7ead1750665161be4**, original audit `/home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-035341-ttie-t036a-common/artifacts/audit`. Config/assets and all100 common trajectory/decision/output/field file hashes match the original freeze. Before the first learned-gradient calculation, provenance.json binds all100 low names, original per-file hashes, fixed legacy step10 and exact float32 legacy-slice hashes. Each later probe rechecks its step10 slice hash and copies those eight raw EV/gamma coordinates unchanged. Gate scores/active/winner/evidence match accepted T036 tensors exactly.

- clip: `1bd3c7172de5b207ceac554f5ab5266166f3b9baccc9af5989bc801016d080ad`
- energy: `c3d1eef9f20af163e1268823cf16d3fdaed315e7723fc94db3b1138ad1336521`
- prototypes: `b4b32dbd96c65dcf606ee38d7450ebf348f5731823503b9c71ba15ec78217ac7`
- gate: `b7458c51466fd89c1bfa8e7c4bf1dd820ddf56e33c12c406f6625ac64361a4ce`

Use exactly **100 new probes, step10 legacy + gain1.75**. The original step10 common-gain channels are ignored. Active common gain is set uniformly1.75 through accepted exp(log2*tanh(raw)); inactive gain stays1. No selected-state recomputation, step0/20/30/40 gradient probes, step/gain sweep, interpolation, optimization, new selection or new cohort. The same100 pairs were already reference-used in T037/T038/T041.

Baselines are bound rather than recomputed:
- selected: `research_log/T041A_result/stage_b/summary.json`, SHA `6c715fe80b79f421901f9ca448a2d5b42e99f3816c523880af7e0070b9079160`
- source: `research_log/T040A_result/stage_b/summary.json`, SHA `50abeaeb1a6d9bfc0c7c5a6cd5752f1bdf39001873b88c0397beaae6c0693799`

T041 selected total positive-dot0.37 and median cosine-0.18584799268346364 are checked exactly. The primary test uses the task's equivalent explicit thresholds total>=0.57 and median>=0.06415200731653636. Source1.75 and the legacy/gain groups are descriptive only. Optional per-image sign-transition and29/71 loss grouping are not implemented; no loss-ID/PSNR/SSIM file is opened.

## Isolation and execution

Stage A run **20260915-133013-ttie-t042a-stage-a**, release20260915-132936-ttie-t042a-step10, executed on A6000 physicalGPU1 in **13.431597s**, exit0. All100 raw states, native outputs, features, energies, masks/gates, g_E and output hashes were frozen **2026-09-15T05:30:37.384859+00:00**, SHA **a5b0799ee370bfd482393fce59895d904008cee9c577d85b385c3ce19dd92c36**. Before freeze: normal decodes0, prior real metric/loss-ID/reference-gradient file opens0, optimizer updates0, selection changes0. Stage A reads baseline path/hash metadata only; actual reference-derived baseline summary files are excluded from its source-file read checks and opened by Stage B only. All tensors/scalars finite; model parameters/normalization unchanged; raw states unchanged during gradients.

Stage B run **20260915-133153-ttie-t042a-stage-b** started **2026-09-15T05:32:00.869550+00:00**. First reference-related baseline open **2026-09-15T05:32:01.172788+00:00**, first normal **2026-09-15T05:32:01.329144+00:00**, both after complete Stage-A freeze. Completed **2026-09-15T05:32:05.212200+00:00**, duration **4.342848s**, exit0 on GPU1. Only the same100 authorized T036 normals under shared/t036a/normal and their lows were decoded. Exact native float32 RGB input handling and isolated float64 mean RGB squared-error gradient are reused.

**All100 Stage-B output hashes and tensors are bit-exact** to Stage A; all100 Stage-A tensor hashes remain unchanged after evaluation/archive. Step10 legacy hashes also remain exact. No reference-dependent state/gain/gate/output/model/decision changes, no optimizer updates/selection changes, no fresh cohort or official-test access.

## Alignment and comparisons

Float64 active-coordinate dot/norm/cosine with norm<=1e-12 degeneracy exactly follows T029/T038. Legacy EV+gamma, gain and their disjoint total are reported separately. Positive-dot fraction and cosine exclude degenerate rows; all100 T042 rows are nondegenerate in each group. Positive dot denotes a locally restorative negative learned-energy gradient for RGB-MSE, not a finite optimizer-step quality guarantee.

| State population | Group | Count / nondegenerate | Median cosine | Positive dot | Median energy norm | Median reference norm | Median dot | Degenerate fraction |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| T041 selected | legacy | 100 / 100 | -0.220698296054 | 0.370000000000 | 0.54647087393 | 0.0559062666342 | -0.00415364162441 | 0.000000000000 |
| T041 selected | gain | 100 / 100 | 0.309086393051 | 0.710000000000 | 0.117068921558 | 0.0122590193774 | 0.00027922770186 | 0.000000000000 |
| T041 selected | total | 100 / 100 | -0.185847992683 | 0.370000000000 | 0.553232202823 | 0.0570827620577 | -0.0033492553553 | 0.000000000000 |
| T042 step10 | legacy | 100 / 100 | 0.686960501945 | 0.940000000000 | 1.32265774784 | 0.078432976892 | 0.0719821132625 | 0.000000000000 |
| T042 step10 | gain | 100 / 100 | 0.535115270581 | 0.780000000000 | 0.092283489479 | 0.0101422455688 | 0.00067015138626 | 0.000000000000 |
| T042 step10 | total | 100 / 100 | 0.685276543900 | 0.940000000000 | 1.324673344 | 0.079249235038 | 0.0729587785733 | 0.000000000000 |
| T040 source | legacy | 7346 / 7248 | 0.770273713456 | 0.796495584989 | 2.53679635582 | 0.0666604063449 | 0.0915295428161 | 0.013340593520 |
| T040 source | gain | 7346 / 7248 | 0.612039101902 | 0.730546357616 | 0.210052226082 | 0.0107693483939 | 0.000695938704835 | 0.013340593520 |
| T040 source | total | 7346 / 7248 | 0.763459378857 | 0.796633554084 | 2.54891968883 | 0.0675701664371 | 0.0919711045495 | 0.013340593520 |

| Comparison | Group | Positive-dot change (pp) | Median-cosine change |
|---|---|---:|---:|
| step10 minus selected | legacy | 57.000000000 | 0.907658797999 |
| step10 minus selected | gain | 7.000000000 | 0.226028877530 |
| step10 minus selected | total | 57.000000000 | 0.871124536583 |
| step10 minus source | legacy | 14.350441501 | -0.083313211510 |
| step10 minus source | gain | 4.945364238 | -0.076923831320 |
| step10 minus source | total | 14.336644592 | -0.078182834957 |

![Legacy-state substitution at fixed gain](T042A_result/legacy_state_alignment.png)

The largest change is in legacy alignment: positive-dot37% to94%, median cosine-0.220698 to0.686961. Gain changes71% to78% and median0.309086 to0.535115. Relative to source1.75, step10 real total positive-dot is14.336645pp higher but median cosine0.078183 lower; this descriptive cross-population comparison adds no gate and is not a claim that real outperforms source on image quality. Source states are correlated canonical bank states; real probes are one fixed state per image.

Independent stdlib replay imports no main alignment/summarize/classify path. It rebuilds active masks from hash-frozen gates, checks all100 identities and step10 provenance, reproduces legacy float32 hashes using struct.pack, recomputes300 gradient group vectors, dot additivity, all aggregates/baseline deltas, verifies the accepted baseline file hashes, and independently recomputes the primary verdict. **2251 scalar checks**, max error **4.440892098500626e-16 <=1e-6**, PASS. Full mean/p10/p90, norm/dot medians and zero/degenerate fractions are in summary.json; all per-state raw/g_E/g_R/masks/scalars/output hashes are in states.json; full output/feature tensors remain in Stage A.

Baseline T0412tests passed19.65s; T0422local tests passed16.50s; T0422server tests passed1.35s. No scientific changes after reference access. One Stage A and one Stage B, no failures/restarts/deviations. Existing NVML warning is nonblocking. Narrow local sparse checkout completed without disk problems.

## Artifacts and next step

Full F archive **289556480 bytes**, SHA **7ce22d5007b8146f16f420ac3205483405e458df113c04b79c10147a24131220**; compact **151373 bytes**, SHA **8a06b0bdd024ba9ca65743178d52bbbcd79567d6a4975490250d801c4e5c36e3**, verified on both server roots and locally. Exact source commands/environment/run logs, provenance/freeze/reference receipts, scalar vectors and plots are retained. Git-exact source/evidence/mailbox recovery including baseline JSON bytes is mirrored in project research_log and home/F shared/t042a. Commands are in stage_a_run/run.sh and stage_b_run/run.sh; independent check is `python research_log/T042A_audit/replay.py research_log/T042A_result`.

Stop after this diagnostic and await research-lead review. The next method choice must address state/feature reliability without assuming a universal deployable step10 policy. No controller, retraining or new experiment is implemented in this cycle.

late real legacy-state effect supported
