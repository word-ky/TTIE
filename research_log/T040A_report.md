# T040-A — DONE: source high-gain tangent deficit not supported / target-domain state shift remains stronger

**SOURCE_REFERENCE_GRADIENT_DIAGNOSTIC_ONLY.** At gain1.75, legacy EV+gamma positive-dot fraction is **0.796495584989**, gain **0.730546357616**, a **6.594922737 percentage-point** legacy-minus-gain gap. Median cosine is legacy **0.770273713456**, gain **0.612039101902**, gap **0.158234611554**. The fixed scientific criterion requires BOTH gaps>=20pp and>=0.25 at gain1.75 only. Verdict: **source high-gain tangent deficit not supported / target-domain state shift remains stronger**. Gain1.50 is descriptive transition evidence and creates no second gate.

## Scope, implementation and bindings

Executed source **b47ac253f6019f778dd245887ede89988b589979**, branch `codex/T040A-high-gain-tangent`, PR [#65](https://github.com/word-ky/TTIE/pull/65), base18a55ca1 after accepted T039 merge1a99369f967e4157623749f4df97634180982b44. Evidence/mailbox SHAs are retained in final delivery. Reuse T039 core helpers and accepted Stage-A/Stage-B loops with only two fixed high probes, accepted-selection binding and gain1.75 verdict scope. No deployable code changes, training, optimizer updates, selection changes, gate/renderer/CLIP/energy edits, new cohort, reference-derived grouping or LOL-v2 access.

All **7346 canonical states /400 banks /80 source-training IDs** are reused in the identical accepted T039/T014 manifest order with duplicates and inactive identities retained. No resampling. Accepted selection SHA **08227ee09f4cd428ee034d1d33cea7827037e853f2fe8b0a7efedcccfecc964c** is checked before probes; bank entries compare equal. Probe only **1.50 and1.75**, exactly **14692** new states; the completed near-range probes are not rerun. Gain1 renderer reconstruction is reused solely as the accepted parity check, without a new gain1 gradient/reference audit.

T014 source manifest SHA **4dbf8c604bc44574a5823ec8e22142c8ac39574b6103c4bbd0a69647f15a2257**; canonical bank manifest **92fd60d01ca0780880d724464c4d0a76ba1721ab5b8a2d054d4035a141673125**; historical CRLF receipt **d4379330bb937bd30345f657f90743f0b23f077e49a93d7fb3c51e6bdfc92de2**; frozen Sobolev checkpoint **c3d1eef9f20af163e1268823cf16d3fdaed315e7723fc94db3b1138ad1336521**. CLIP/prototype identities, normalization buffers and every deployed source hash are retained in freeze/source_binding. All400 banks' bank.pt/bank_images.pt/bank_decisions.json match accepted hashes; original source-supervision files are not opened by Stage A. Normalization exactly reproduces all7346 cached feature statistics. Raw legacy EV/gamma/gates remain exact. Shared gain uses accepted CommonRegion2 raw parameterization exp(log2*tanh), inactive gain1.

Frozen T039 gain1.25 baseline summary SHA **6607b88f26b17cda18e6c34866d2f56b555e5164bdb8a6753dc8c3e607c43fdf**. Gain positive-dot **0.907836644592**, median cosine **0.862191500210** are copied from accepted evidence, along with legacy and total values. No baseline recomputation or refitting occurs. Full copied baseline is in T040A_frozen_baseline.json and Stage-B summary.

## Isolation, execution and verification

Stage A uses only retained source-bank tensor inputs; PIL source-image opening is prohibited. A clean-condition bank is used through its stored input tensor role, never as reference supervision. Gain1 reconstruction maximum pixel error **2.682209014892578e-07 <=1e-6**; cached feature drift **4.050135612487793e-05**, recorded under the accepted T039 rule without a feature-equality gate. No cached-feature bit-exact claim.

GPU1 Stage A **20260915-105344-ttie-t040a-stage-a**, release20260915-105308-ttie-t040a-high-gain, completed in **894.600896s**. All14692 learned gradients, states, features, energies, masks and output hashes froze **2026-09-15T03:08:48.332442+00:00**, freeze SHA **639f26ffe1a610fea4f3e547152a8edb281912b49d0645ec0c99d1f24277e2da**, source/JPG opens0. Stage B started **2026-09-15T03:10:08.440465+00:00**, first source JPG **2026-09-15T03:10:09.477345+00:00**, completed **2026-09-15T03:12:41.471966+00:00**, duration **153.032160s**. Stage-B run ID is recorded in run metadata/delivery. Both stages exit0 on A6000 physical GPU1.

Stage B opens only80 authorized source-training JPGs under shared/t008/val2017 using exact accepted T014 load_image: float32 RGB, shorter side320, CPU bicubic antialias and clamp. Isolated reference objective is float64 mean RGB squared error. All **14692 output hashes match exactly**, all400 Stage-A files unchanged. Zero optimizer updates/selection changes; finite gradients/scalars; raw states and frozen models unchanged. No LOL-v2 low/normal or official-test image is accessed.

Alignment is the unchanged T029/T038 active raw-coordinate convention: legacy EV+gamma, common gain, total. Dot and norms are float64; norm<=1e-12 makes the cosine undefined and excludes that row from cosine/positive fraction. Inactive rows remain in counts and zero/degenerate fractions. Positive dot means a negative learned-energy gradient is locally restorative for RGB MSE, not a finite Adam/projection guarantee.

| Gain | Group | Count / nondegenerate | Median cosine | Positive dot | Median energy norm | Median reference norm | Median dot | Degenerate fraction |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| T039 frozen 1.25 | legacy | 7346 / 7248 | 0.954278283516 | 0.965921633554 | 2.59919427848 | 0.074690669055 | 0.167884547461 | 0.013340593520 |
| T039 frozen 1.25 | gain | 7346 / 7248 | 0.862191500210 | 0.907836644592 | 0.493454178511 | 0.0321444272161 | 0.00938045858759 | 0.013340593520 |
| T039 frozen 1.25 | total | 7346 / 7248 | 0.916273820714 | 0.966887417219 | 2.66521904581 | 0.0815803350878 | 0.178854153489 | 0.013340593520 |
| 1.5 | legacy | 7346 / 7248 | 0.896870570264 | 0.894177704194 | 2.59905999662 | 0.0680379990292 | 0.133832902889 | 0.013340593520 |
| 1.5 | gain | 7346 / 7248 | 0.774447386157 | 0.839955849890 | 0.374453815203 | 0.0205910473398 | 0.00365854684546 | 0.013340593520 |
| 1.5 | total | 7346 / 7248 | 0.876518919638 | 0.894867549669 | 2.63510845831 | 0.0712381540426 | 0.139025112166 | 0.013340593520 |
| 1.75 | legacy | 7346 / 7248 | 0.770273713456 | 0.796495584989 | 2.53679635582 | 0.0666604063449 | 0.0915295428161 | 0.013340593520 |
| 1.75 | gain | 7346 / 7248 | 0.612039101902 | 0.730546357616 | 0.210052226082 | 0.0107693483939 | 0.000695938704835 | 0.013340593520 |
| 1.75 | total | 7346 / 7248 | 0.763459378857 | 0.796633554084 | 2.54891968883 | 0.0675701664371 | 0.0919711045495 | 0.013340593520 |

| Gain | Legacy-minus-gain positive-dot pp | Legacy-minus-gain median cosine | Gain positive-dot change vs frozen1.25 pp | Gain cosine change vs frozen1.25 |
|---|---:|---:|---:|---:|
| 1.5 | 5.422185430 | 0.122423184107 | -6.788079470 | -0.087744114053 |
| 1.75 | 6.594922737 | 0.158234611554 | -17.729028698 | -0.250152398308 |

![High-gain source tangent alignment](T040A_result/high_gain_alignment.png)

All per-probe scalars, masks reconstructed from gates and raw g_E/g_R vectors are retained in states.json/vectors.json; full means/p10/p90 and energy/reference zero fractions are in summary.json. Independent stdlib replay imports no T029/T038/T039/T040 alignment, summarize or classify helper, checks all bank/state/gain ordering and masks, and recomputes the gain1.75 verdict. **322753 scalar checks**, all14692 probes/44076 groups, max absolute error **1.7763568394002505e-15 <=1e-6**. Baseline tests2passed; T040 local tests2passed19.31s. Server focused tests2passed1.33s; exact output is retained in Stage-A train.log.

## Failures, limits and archive

Before any GPU experiment, a broad new-worktree checkout filled D. Git sparse-checkout plus restoration of partial checkout files recovered the clean worktree and1.64GB free; no existing project evidence was altered. Automatic approval rejected an initial combined process-stop/removal command, so that removal never ran; Git managed repair succeeded. Two baseline invocations during recovery passed but reported unusually long wall times3026.08s/3038.61s. Preparation therefore exceeded the approximate one-hour budget; experimental scope remained the single fixed audit. Research-lead T040-A-EXEC commit eed82b65 was observed while the same Stage A was running (~58% complete); it accepts the exact executed source and leaves every scientific setting unchanged. No second experiment was launched. The existing NVML warning is nonblocking. No scientific threshold/gain/state change.

Full F archive **724561920 bytes**, SHA **fd9e09ae639a8ffb9cce4ab17c7676f5ed13129ee876dfd117ba5e3644fd61fe**; compact evidence **5864443 bytes**, SHA **df2006c0415d6d0f80e4343281d8c0dbcbc38665bb5c4708ce8f071c12a8d2c7**, both server roots and local. Original T014 bank assets remain at their bound source run. Exact commands/env, state selection, run logs and output receipts are preserved; final Git-exact recovery package mirrored locally and to home/F shared/t040a.

Both legacy and gain alignment decline as common gain increases. Gain positive-dot falls17.729pp and median cosine0.250152 from frozen1.25 to1.75, but the corresponding within1.75 legacy-minus-gain differences remain only6.594923pp/0.158235. This does not meet the prescribed gain-specific deficit gate, and it does not mean the high-gain source field is unchanged or uniformly reliable.

This is a source-training local derivative diagnostic, not a held-out qualification, proof of a deployable repair or proof that retraining would help. T038's real loss-subset and this all-source-state population are different; domain and trajectory effects are not experimentally separated here. Stop and await research-lead judgment regardless of verdict. Do not retrain or add a controller in this cycle.

source high-gain tangent deficit not supported / target-domain state shift remains stronger
