# T028-A — T026-A-family reference-oracle audit

**substantial within-family headroom** under the predeclared rule. Paired oracle
minus accepted T026-A PSNR has mean **+6.339115361dB** and median
**+5.771085393dB**, exceeding the required+2dB mean and+1dB median thresholds.
This is a quarantined reference-assisted diagnosis, not deployable enhancement,
method promotion, official-test performance or a certified global optimum.

## Fixed family, provenance and reference ordering

Scientific source `80006657e9b270644f8e0ad9fdf1cb6309966807`, release
`20260914-163059-ttie-t028a-oracle`. Accepted T026-A source
`b2359721c89db732d17e03be273e0bdb71bb377a`, merge
`51f84a9d96880bca8e908c9b3cdff496d7277e39`, accepted run
`20260914-113853-ttie-t026a-gamma05`. Seven renderer/bounds/metric source blobs
are identical to accepted T026-A, and deployed source hashes are unchanged.

The split/order is the same100 validation pairs, SHA256
`b88c8347005984b5523b117b52c0c068672fe172eb7c9aa5b60102d350e2d85b`.
No official-test filenames/content are enumerated or decoded in this task.
`T028A_preflight/preflight.json` binds accepted freeze/config/metrics and every
decision/output/trajectory/low-image hash. All100 selected outputs and physical
grids reconstruct with **exactly zero maximum error**. Both predeclared starts
(identity and frozen T026-A selected raw state) are persisted before references.

Preflight run `20260914-163126-ttie-t028a-preflight` completed at
`2026-09-14T08:31:39.695754+00:00`, with exactly100low opens and zero normal opens.
Preflight SHA256:
`84212f1b19950db7b51fd746d4f546af35330cc565f1376c76bfe284bd2e6f16`.
Task-specific reference deployment only began at
`08:32:39.325379+00:00`, completing at`08:32:39.429927+00:00`.
Named validation references are copied/hash-verified without decoding during
deployment, then opened only by the isolated oracle and postfreeze evaluator.
Other-task references outside the preflight allowlist are not claimed absent.

Exact accepted family: frozen gate/Region2, active darkEV[0,2], brightEV[-.5,0],
gamma[.5,1.25], inactiveidentity, unchanged float32 renderer/projection. Reused
T025-A Adam lr.05, betas(.9,.999), eps1e-8, weight_decay0; **500updates/start**
against full-frame RGB reference MSE withfloat64 accumulation. No learned-energy
or CLIP computation, additional operator, retraining or alternate start.
Across both saved histories choose minimum MSE, earliest step, then identity for
any remaining exact tie. Full501raw states and MSE values per start are retained.

## Completed experiment and quantitative result

The sole A6000physicalGPU1 oracle run
`20260914-163238-ttie-t028a-oracle` completed100images,200starts,100000updates,
exit0. All raw states, gradients, outputs and objective values are finite and
inside the exact action family; all200histories have501states. All100outputs and
histories froze at`2026-09-14T08:51:24.878347+00:00`. PSNR/SSIM evaluation ran
only after that global freeze and completed at`08:52:08.400219+00:00`.

| Output | Mean PSNR | Median PSNR | Mean RGB-SSIM | Median RGB-SSIM |
|---|---:|---:|---:|---:|
| Raw | 8.109722672 | 7.600161541 | 0.160022843 | 0.138977265 |
| Accepted T026-A | 11.120876417 | 10.607319299 | 0.373791825 | 0.367924983 |
| REFERENCE_ORACLE_ONLY | 17.459991778 | 16.342409074 | 0.431715830 | 0.480401006 |

| Paired oracle minus T026-A | Mean | Median | p10 | p90 |
|---|---:|---:|---:|---:|
| PSNR dB | +6.339115361 | +5.771085393 | +2.393411790 | +11.658975864 |
| RGB-SSIM | +0.057924005 | +0.050326644 | -0.020611525 | +0.144611667 |

All100 images improve in PSNR; smallest gain+0.722664899dB, largest
+16.124403256dB. **18/100 images lose SSIM**; worst SSIM delta
`-0.104871433`. Full per-image values/deltas and worst cases are preserved, not
hidden by mean improvement. The oracle optimizes MSE, not SSIM.

Winning starts: **87 T026-A selected,13 identity**. Winner-step histogram:
208:1,298:1,362:1,461:1,499:2,500:94. Identity-start best-step histogram:
192:1,208:1,461:1,500:97. Selected-start best-step histogram:
202:1,298:1,362:1,487:1,499:2,500:94. All200starts still execute500updates.

Boundary hits at oracle winners, physical tolerance1e-6:

| Coordinate group | Lower hits | Upper hits | Either bound |
|---|---:|---:|---:|
| Active EV,392 coordinates | 7 | 75 | 82 |
| Active gamma,392 coordinates | 0 | 0 | 0 |
| Inactive EV,8 identity coordinates | 8 | 8 | 8 |
| Inactive gamma,8 identity coordinates | 8 | 8 | 8 |

Inactive lower/upper coincide at identity; those counts are separate from active
bound saturation. No bound expansion was used. Since94winners select500, the
finite search is not a convergence certificate. It establishes demonstrated
within-family reachable improvements under this fixed oracle, not an exact
mathematical ceiling. No extra budget/run is authorized by this observation.

## Validation and runtime

- Baseline4tests passed24.75s; local focused oracle test1passed17.85s; remote
  focused test1passed2.56s. Tests cover earliest step0 retention, projection,
  inactive pixel identity and saved-state/history alignment.
- All100 selected-output/grid reconstructions exact before any reference pixel.
- Postfreeze audit checks all200finite501-state histories, bounds, minimum-MSE
  winners/earliest ties, selected raw-state equality and nonworse reference MSE.
- Accepted T026-A/raw PSNR and SSIM reproduce within1e-11. Independent separable
  SSIM/Torch PSNR maximum absolute error is`7.105427357601002e-15`.
- Independent Python statistics/linear quantiles from CSV agree with JSON and
  NumPy to1e-12; classification is unchanged.
- All300frozen output/state/history hashes are verified remotely; all200fetched
  compact state/history hashes and deployed oracle source bytes verify locally.

Environment: NVIDIA RTX A6000physicalGPU1, PyTorch2.4.0+cu121, CUDA12.1, seed7,
TF32off. Per-image runtime includes hashes/load/two500update starts and saved
artifacts, excludes preflight and later scoring: mean11.177044569s,
median11.343406965s, p9511.671832463s, total1117.704456916s. No competing baseline
or deployable trajectory was executed.

Observed setup issue: sparse checkout initially omitted the older oracle source
needed by baseline tests; adding the tracked folder restored the test. An
unsupported sparse-add flag was corrected. Existing NVML and Python deprecation
warnings did not block CUDA execution. No scientific run failed or repeated,
no numerical tolerance or task setting was relaxed.

## Artifacts and interpretation

Code/protocol/source binding: `research_log/T028A_oracle/`,
`T028A_protocol.md`, `T028A_source_binding.json`.
Preflight/starts/reference-deployment: `T028A_preflight/`.
Compact full histories, states, per-image metrics, aggregate/metric receipts,
freeze hashes, config and commands/logs: `T028A_evidence/`.
No oracle quantities are written into Ours, any head/gate/selector/training
input or official-test path; diagnostic artifacts remain REFERENCE_ORACLE_ONLY.

Full outputs/source/preflight/runs backup:
`/media/wenchang/F/wjq/TTIE/shared/t028a/T028A_execution.tar`,295137280bytes,
SHA256`be8ed83c85788fd1bed381d2f308fa8ac9557c7a0bbe3fc2e5997dd7805eb8a2`.
Compact bundle SHA256
`e089d85b848a58f89c1035e9af5edd5a93de61e8fe4a376d036cf0358d89a9b6`.
Full output tensors remain remote/backup, not Git. Validation references can be
recovered from the previously bound canonical LOL-v2 archive using the frozen
split and per-file hashes; this task did not reopen official-test members.

The evidence answers the requested diagnosis: the promoted action family still
contains substantially better reference-assisted states than the learned
trajectory reaches. It does not identify a deployable way to find those states.
Retain T026-A as the promoted deployable candidate, preserve the sealed official
test, and return the diagnosis to the research lead. No follow-on experiment.

substantial within-family headroom
