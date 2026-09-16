# T058-Z DONE — T058 derivative verifier unresolved

The authorized single-cast-elided CPUfloat64 shadow runs, the exact same-input cast-back identity passes, and both primal/reverse consistency criteria pass. **The predeclared clamp one-sided-secant interval criterion fails.** The float64 reverse derivative .00634846690811682 exceeds the expanded upper bound .0056557882718651425 by `0.0006926786362516776`. No rescue or extension was performed. T058-A remains PARTIAL; no source-readiness result.

Source `5b9759dd8cfe7ac9339689422b7ae22a979740e5`; branch `codex/T058Z-cast-isolation`; PR https://github.com/word-ky/TTIE/pull/89; authorization `0cdaa94399f64739cb8bd208f4205e88ab68df0f`. Exact original T058-A source aa22caacd42906ba36063a1a2560600ba2370897 and historical stopped evidence e7953a202112baf647654411626e743865ae8f25 preserved. All161 deployed files,8 T058-A artifacts and7 T058-Y artifacts match before/after. Donor first16 selection SHA `4e18e346478421ebbbd192fdbe6c5a861cdc0a45761ecf7a3becb7afeec079c2` retained for provenance; actual execution was **only canonical index3**, bank1/state2/image36660. No other15-row computation.

Only verifier-local `research_log/T058Z/shadow.py` reproduces the accepted energy.features expression in the same order and omits its final .float(). No ttie/model/CLIP/head/checkpoint/preprocessing/backend changes. Same CPUfloat64 scores/objective/grid fed both the unmodified accepted function and the verifier expression at v=0. Accepted output is bitwise equal to phi64.float(): **true**. Both28-D vectors and every concatenation input dtype are saved in the row receipt. All concatenation inputs/features are CPUfloat64; energy/output/features are allfloat64. This isolates the one authorized terminal cast from CPU/GPU/scorer-precision differences.

| Criterion | Observed error | Allowed tolerance | Margin | Result |
|---|---:|---:|---:|---|
| Same-input cast-back identity | 0 bit differences | exact | exact | PASS |
| E64 versus E32 | 1.28349222805468e-07 | 0.000959264208714278 | 0.000959135859491473 | PASS |
| d32_rev versus d64_rev | 6.39119884047329e-08 | 5.17423345405841e-05 | 5.16784225521794e-05 | PASS |
| d64_rev inside expanded h=.0005 one-sided interval | upper excess 0.000692678636251678 | expansion 1.46969338162336e-05 | -0.000692678636251678 | FAIL |

E32 `-4.796320915222168`; E64 `-4.796321043571391`; d32_rev `0.006348530820105225`; d64_rev `0.00634846690811682`. Original historical reverse .006348532158881426 and original h=.001 central .003814697265625 remain unchanged; the new float32 reverse differs by only 1.33877620100975e-09. No tolerance tuning.

| h | E64(+h) | E64(-h) | central | positive-side secant | negative-side secant |
|---|---:|---:|---:|---:|---:|
| 0.004 | -4.7963087123973756 | -4.796343062102977 | 0.004293713200209659 | 0.003082793503805803 | 0.005504632896613515 |
| 0.002 | -4.796314881398227 | -4.796332229775275 | 0.004337094261952501 | 0.003081086581868675 | 0.005593101942036327 |
| 0.001 | -4.7963179633488595 | -4.796326684314295 | 0.00436048271756917 | 0.0030802225312598353 | 0.005640742903878504 |
| 0.0005 | -4.7963195036774575 | -4.79632386411706 | 0.004360439602280053 | 0.0030797878665111966 | 0.005641091338048909 |

Boundary-directional RGB count3042; minimum nonzero active boundary distance2.115964889526367e-6, matching T058-Y. At h=.0005 the one-sided interval is `[0.0030797878665111966, 0.005641091338048909]`, expansion `1.469693381623364e-05`, expanded interval `[0.003065090932694963, 0.0056557882718651425]`. Reverse lies above it. Both smallest-step central secants remain near .00436044 rather than .00634847 in float64: a discrepancy remains after eliminating the terminal feature float32 cast. Thus this result does not establish a pure float32-quantization explanation. Clamp nonsmoothness is present, but the **specified** clamp-convention acceptance rule is not satisfied. This does not by itself prove that framework reverse AD is mathematically incorrect; the task's numerical credibility criterion remains unmet.

No Python exception occurred in the numerical path; the explicit failure receipt is `FLOAT64_FD_CRITERION_FAILURE`. There is no traceback to invent. Source/scorer/head/checkpoint/detail/legacy and nonpersistent shadow hashes agree before/after. Zero optimizer updates, persistent scientific-state changes, source-clean/JPG opens, StageB, target-domain/official-test access. No additional cast removal, h, tolerance, backend, precision, device route or rerun.

Local4tests passed5.07s; server4tests passed1.56s; compilePASS. Sole run `20260917-074144-ttie-t058z-cast`, release `20260917-074118-ttie-t058z-cast`, originalGPU1 + verifierCPUfloat64, elapsed `29.788405498024076` seconds, exit0 records the unresolved verdict rather than acceptance. All4 fixed perturbations completed exactly once; no full scientific audit. Run start 2026-09-17T07:41:49+08:00; finish 2026-09-17T07:42:23+08:00. Exact commands are retained in run.sh.

Archive `{"path": "/home/wenchang/asdasdsad/wjq/TTIE/shared/t058z/T058Z_evidence.tar.gz", "backup": "/media/wenchang/F/wjq/TTIE/shared/t058z/T058Z_evidence.tar.gz", "bytes": 68877, "sha256": "0918382ac0a0e2ecfb140b9cd7c6b290277230e70db0c6db6437bcfa60e7fcb7", "source161_unchanged": true, "historical8_unchanged": true, "prior_verifier7_unchanged": true, "files": {"train.log": "96801ebfcb0e9425604afc94f6cd77601271ae3ba5a5ba81938975ab87484c15", "meta.json": "9009d6060a36394eff790520dc7b5e1c88a01d49218e5b1b9ffcb553fba41fd8", "run.sh": "a3de8e752e5d0cf450bb190b50d0a88f477435f3c7cc565027707d93c39b1503", "artifacts/T058Z/receipt.json": "00a2633c6535f55ffbf954b71f972aa0171255b67ca79c02be575873f618c9e3", "artifacts/T058Z/selection.json": "4e18e346478421ebbbd192fdbe6c5a861cdc0a45761ecf7a3becb7afeec079c2", "artifacts/T058Z/states.json": "0519bcbe81f3d8206064abc23ebf6d523a28037ac6f3514a09f65faa356aceb9"}}` verifiedhome/F/local, all6 constituent files exact. Row receipt stores featurevectors/dtypes/identity, primals/derivatives/criterion margins, all raw perturbation energies/secants, boundary diagnostics and original historical FD. Scientific data and original failure receipts untouched.

Recommendation: research lead should adjudicate the remaining row3 reverse-versus-directional-secant discrepancy and the failed interval criterion before continuation. Stop after fixed row3 FD criterion failure; await research-lead adjudication. No newcasts/h/tolerance/backend/other15rows/fullT058A/StageB/JPG/T059.
