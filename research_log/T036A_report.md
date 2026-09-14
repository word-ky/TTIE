# T036-A — DONE: materially positive

On the single frozen fresh100-pair training-development cohort, common-gain minus exact T026-A mean **+0.9392570676866118 dB PSNR /+0.008356044474490985 RGB-SSIM** passes the predefined >=0.30 dB and >=0 SSIM joint gate. Median paired gains are **+0.6439381963536586 dB /+0.005337860542672525 SSIM**. This is target-free adaptation followed by held-out-for-this-cycle evaluation, not a reference oracle. The aggregate success does not establish per-image safety: PSNR declines on29/100 images and SSIM on40/100; the worst case loses5.61447dB and0.11854SSIM.

## Freshness and information boundary

Main35309c0d, accepted T035 mergeceecd091. The accepted T032 ledger and its bound prior receipts account for216 reference-used pairs: originalvalidation100, T023 real-source16, T030 qualification/T031 diagnostic100. T032 actual evaluation receipt adds100 more. T033/T034/T035 actual normal-open lists were checked to equal originalvalidation100, adding no new pair. Therefore316 of689 official training pairs are excluded, leaving373 eligible. The new cohort is exactly the first100 after sorting SHA256(normalized relative low path), with no file hash concatenation or alternate seed. Manifest SHA **279c74b335999d6631d436c79ea80e9e2cccfc4b97cb7c0b992829d04cfaf40b** was frozen before any task image payload access. Exclusion SHA **b52901e774808ac3dcc7f5f834a11e7a12c07df5fb9d933cf7c9eeb6700c1537**. Candidate-level exclusions, accepted receipt hashes and prior ledger are retained in `T036A_cohort/`.

Historical dataset indexing read encoded hashes/PNG IHDR dimensions, not normal pixels; freshness concerns project reference-pixel use documented in accepted receipts. The historical T023 source-receipt hash in the T032 ledger represents a CRLF working copy; that exact representation is explicitly reconstructed for this one receipt check, with content unchanged. T033's normal-open records are path/UTC dictionaries and are parsed accordingly. Two early cohort-preparation attempts exposed these metadata-format differences and stopped before writing a cohort or reading images; the final manifest was created once after both were handled.

**These100 pairs are now reference-used development data and must be excluded from future fresh qualifications.** No official-test path or image is used in T036. No T035 per-image oracle gain, metric, state or reference gradient is supplied to either adaptation method. The choice of common gain was motivated by earlier development diagnostics; the current paired evaluation is a single fresh qualification of that fixed choice, not official-test/SOTA evidence.

## Frozen implementation, assets and preflight

Source **f80cea4c9d8186e0c4a0404b28ccd58c5e1b5678**, branch `codex/T036A-common-gain-qualification`, PR #61. Accepted T026 source **b2359721c89db732d17e03be273e0bdb71bb377a**: all17 donor blobs match their accepted hashes. Existing T026 modules are unchanged. New `ttie/common_gain.py` adds one raw gain per existing Region2 cell and expands it identically RGB at the existing WB position; physical gain `exp(log(2)*tanh(raw))`, identity1 and bounds[.5,2]. Existing hard masks, inactive output bypass, EV/gamma bounds/projection and renderer order remain unchanged. `common_gain_ttt.py` differs only in model/box wiring; AST equality checks prove both the complete trajectory function and energy-evaluation function match accepted T026. The frozen energy still consumes the exact original28 features; gain influences it only via the rendered image and existing CLIP evidence. No energy/head training or new input feature is introduced.

Both methods begin at identity with a fresh Adam optimizer, lr.03,40updates, same minimum predicted-energy checkpoint selection with earliest tie. API arguments are image tensor, scorer, frozen gate receipt/head, basis and budget, with no reference path. The runner's only image decoder is restricted to the selected low-image paths. Original target mutation/withholding tests pass; the new trajectory is AST-identical except class/box wiring.

Exact retained assets: T014 energy SHA `c3d1eef9f20af163e1268823cf16d3fdaed315e7723fc94db3b1138ad1336521`; CLIP `1bd3c7172de5b207ceac554f5ab5266166f3b9baccc9af5989bc801016d080ad`; prototypes `b4b32dbd96c65dcf606ee38d7450ebf348f5731823503b9c71ba15ec78217ac7`; gate `b7458c51466fd89c1bfa8e7c4bf1dd820ddf56e33c12c406f6625ac64361a4ce`. Actual asset-file entries match accepted T026 execution config and hashes verify before/after qualification. All135 staged source/input files were bound successfully before inference.

Low-only preflight `20260915-035138-ttie-t036a-preflight`, completed **2026-09-14T19:52:41.345482+00:00**, SHA **aae37435b3f51d843c787d6c3ae3be6f438845808805ca4e2fea7fc7ddc15cbf**, exited0. Identity common gain matches T026 for all100 original validation lows at identity/accepted selected states, including accepted selected outputs: maxabs **0.0**. The first original T026 image was also run through its actual40step low-only procedure: selected step matches and output maxabs **8.940696716308594e-08**, within1e-6. Nonactive gains remain exactly1. Preflight normal decodes0; prior selected states are used only for renderer regression, never as fresh-cohort initializations.

Local accepted baseline tests **3passed17.24s**; new focused tests **3passed17.07s**; server combined **6passed3.64s**. Tests cover exact trajectory/energy reuse, API shape, active/inactive rendering, one gain coordinate, unchanged energy selection and bounded states. Existing baseline checks also cover reference mutation/withholding and exact saved selected tensors.

## Execution and pre-reference freeze

Sole A6000 physicalGPU1 qualification **20260915-035341-ttie-t036a-common**, release `20260915-035113-ttie-t036a-common`, exited0. Both methods each completed **4000 updates/4100 states** across100images, with identical gates. All200 outputs, decisions, score histories, raw/EV-gamma fields and explicit full fast fields were frozen before reference deployment. Every fresh image was decoded once, as a low input; normal decodes0. Full saved-state replay checked all800 files, all200 min-energy selections, finite/bounded states/images, exact field replay and inactive gain identity.

Outputs froze **2026-09-14T20:02:00.637047+00:00**, SHA **46e667ded785e4f3f8ba341d40d00a7e02ca652e2778fcc7ead1750665161be4**. A separate reference deployment began **2026-09-14T20:02:56.880506+00:00**, after successful full replay; it wrote the expected freeze SHA outside the audit before reading any selected normal member. The separate evaluator binds that receipt and freeze before opening references. CPU evaluation run **20260915-040317-ttie-t036a-eval** completed **2026-09-14T20:03:51.777280+00:00**, exit0. Exact T026 convention: native fullRGB float32 pixel values promoted tofloat64, no crop/resize/Y conversion/PNG quantization; same RGB-SSIM implementation. No per-image setting or stopping choice reads a reference.

## Absolute and paired metrics

| Method/comparison | Mean PSNR | Median PSNR | Mean RGB-SSIM | Median RGB-SSIM |
|---|---:|---:|---:|---:|
| Exact T026 baseline | 10.290783069326 | 9.670497330119 | 0.337946272178 | 0.298042116960 |
| Common gain | 11.230040137013 | 10.783092414351 | 0.346302316652 | 0.321560144815 |
| Paired common minus baseline | 0.939257067687 | 0.643938196354 | 0.008356044474 | 0.005337860543 |

Win/equal/loss PSNR **71/0/29**, SSIM **60/0/40**. Worst image `Train/Low/low00559.png` (index33) drops from20.1357529133dB/.7221431833 to14.5212843239dB/.6036042710, paired -5.6144685894dB/-.1185389124; both selectstep40. No rescue or retuning was attempted. Baseline selects40 for86images; common selects40 for99images and36 for1. Full histograms and all per-image values are retained in `T036A_result/audit/summary.json` and `metrics.csv`. Frequent budget-end selections are reported without extending the budget.

## Selected fast-state distributions

391 active/9 inactive region instances per method. Full active/inactive/all groups by channel/region, includingp05/p95 and all2400 scalar records, are in summary/bound_values. Physical-bound hits use1e-6 tolerance and original region-specific EV/gamma boxes.

| Method | Channel | Region | Count | Mean | Median | Min | Max | Lower/upper hits |
|---|---|---|---:|---:|---:|---:|---:|---:|
| baseline | ev | all | 391 | 0.668195 | 0.734203 | -0.500000 | 1.678128 | 33/14 |
| baseline | gamma | all | 391 | 0.775511 | 0.653804 | 0.564647 | 1.250000 | 0/59 |
| baseline | gain | all | 391 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0/0 |
| common | ev | 00 | 97 | 0.232275 | 0.345546 | -0.500000 | 1.266889 | 12/0 |
| common | ev | 01 | 99 | 0.762648 | 1.153599 | -0.500000 | 1.462385 | 17/0 |
| common | ev | 10 | 98 | 1.005286 | 1.318785 | -0.500000 | 1.590316 | 1/3 |
| common | ev | 11 | 97 | 0.354990 | 0.402067 | -0.500000 | 0.982443 | 4/15 |
| common | ev | all | 391 | 0.590754 | 0.562545 | -0.500000 | 1.590316 | 34/18 |
| common | gamma | 00 | 97 | 0.847225 | 0.677124 | 0.563199 | 1.250000 | 0/19 |
| common | gamma | 01 | 99 | 0.676233 | 0.641343 | 0.592592 | 1.247120 | 0/0 |
| common | gamma | 10 | 98 | 0.806277 | 0.650425 | 0.575079 | 1.250000 | 0/22 |
| common | gamma | 11 | 97 | 0.776905 | 0.659711 | 0.607538 | 1.250000 | 0/15 |
| common | gamma | all | 391 | 0.776222 | 0.657658 | 0.563199 | 1.250000 | 0/56 |
| common | gain | 00 | 97 | 1.105991 | 1.055936 | 0.557085 | 1.776598 | 0/0 |
| common | gain | 01 | 99 | 1.348693 | 1.522436 | 0.568458 | 1.795160 | 0/0 |
| common | gain | 10 | 98 | 1.111538 | 1.130850 | 0.570313 | 1.772754 | 0/0 |
| common | gain | 11 | 97 | 1.539634 | 1.670165 | 0.582571 | 1.784919 | 0/0 |
| common | gain | all | 391 | 1.276412 | 1.439302 | 0.557085 | 1.795160 | 0/0 |

Common active gain mean1.276411564, median1.439302087, min.557085216, max1.795160294, lower/upperhits0/0. All9 inactive gains remain exactly1 throughout trajectories, as do all baseline gains. Common active EVhits34/18 and gamma0/56 versus baselineEV33/14 and gamma0/59. Inactive EV/gamma coincident identity constraints are counted separately.

## Runtime, independent checks and artifacts

Torch2.4.0+cu121/CUDA12.1/Python3.12.12. GPU trajectory timing begins with the CUDA low tensor ready and includes both endpoint synchronization calls but excludes disk serialization/model load. Baseline mean/median/p95 **2.408288284272 /2.430968358996 /2.512348512225 s**; common **2.404914512566 /2.442340960988 /2.521268316175 s**. Total measured trajectory time240.828828427s baseline +240.491451257s common. Small timing difference is not a speedup claim.

Independent dot-product PSNR and separate RGB-SSIM implementation agree across200 scored outputs to maxabsoluteerror **2.1316282072803006e-14**. Local stdlib replay independently recomputes all metric/runtime/distribution summaries, paired deltas, histograms and fixed verdict: maxdifference **1.7763568394002505e-15**; all600 retained decision/trajectory/fast-field hashes match frozen evidence. Full output tensors were checked onserver and remain there/F; compact local evidence does not contain them.

Full archive `/media/wenchang/F/wjq/TTIE/shared/t036a/T036A_execution.tar`, **608348160 bytes**, SHA **65f765766a05f25e994b7d5b65c38a443376181cf3a745872d601f4fa5ee2e71**. Compact **5198372 bytes**, SHA **3f6d78424e8cb48f2c3966bd7e074f79faf6b03bdb6a74421858da1a779c2cba**, verified on both roots and locally. Exact command/config/preflight/run/evaluation logs and external deployment receipt are retained. Report/delivery/recovery summaries are mirrored to the outer project research_log.

Operational deviations only: two early metadata-format parsing failures before cohort creation; preflight tmux-launch SSH closed255, then no session/log was confirmed and the same prepared run.sh was started once; archive SSH attempt closed255, no archive receipt/process existed, retry succeeded. No scientific inference/evaluation failure, restart, design change or additional cohort. Known NVML/dependency warnings were nonblocking. Ddisk pressure was resolved using reversible NTFS compression inside the project, stopped after about1.8GB free; no artifact was deleted or moved out of the project.

Return the positive fresh qualification together with29PSNR/40SSIM regressions and the severe worst case to research lead. Stop after this verdict; leave scientific promotion/state updates to the owner. No official-test or external baseline run, learned-energy retraining, per-channel WB, sweep or oracle transfer. These100 are now excluded from future fresh cohorts. PR #61 awaits review without self-merge.

materially positive
