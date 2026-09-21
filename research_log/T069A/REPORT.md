# T069-A — DONE / PROJECTION_PRESSURE_SIGNAL_ABSENT

The only unsafe T067-C selected endpoint has zero final-transition clipping: R_proj=0, below development T99_proj=0.7311988023009769. Unsafe flagged0/1; safe flagged1/99. All200 CommonBox transition checks and200 endpoint GPU render checks pass exactly. Close this exact endpoint projection-pressure statistic; no guard is implemented.

Source: `d19973d9058844cc1ec54639f61e77efc6cbb13d`; branch: `codex/T069A-projection-pressure`; authorization: `366b44389d925cd02f3195d0278dc97c4dc37c21`.

Exact final transition uses states[k-1][:,:2], pre_box[k-1][:,:2], states[k][:,:2], i.e. eight EV/gamma coordinates across all four nodes and no gain channel. Float64 ||proposal-end|| / max(||proposal-previous||,1e-12), with specified k0/zero-proposal score0 only. Fixed lambda=.875/rho=.9857470621423519/model/optimizer/box retained; no optimizer rerun/model fit. Full prior-state/pre-box/end/bounds hashes recorded per row.

Target-free development100scores produce exact nearest-rank T99_proj=sorted_R[98]=0.7311988023009769; 1 development endpoint exceeds it. Development freeze UTC `2026-09-21T03:57:20.760257+00:00`, SHA256 `babc3f6a508168f0f95c21d145857e985df7aabaf8d40134077130f2adaf8b67`; no development reference quality.

Transfer100endpoint score/proposal/state/output/bounds/rule/input freeze UTC `2026-09-21T03:57:21.913543+00:00`, SHA256 `c61ffdaca353807f0cc5abeb56afe397a0bd33efe284dfce10ef7348a4db8efd`, reference_reads=0. First transfer quality-read boundary UTC `2026-09-21T03:57:21.924519+00:00`; only afterward hash/read accepted label artifacts and join T067-C endpoint margins. T066-B same-state labels independently crosscheck.

| Group | Endpoints | Strict R_proj>T99_proj |
|---|---:|---:|
| Unsafe | 1 | 0 |
| Safe | 99 | 1 |

| Index | Safety | k_FS | k_lambda | Norm proposal | Norm clip | R_proj | Descending rank | Percentile | Margin vs T026 | Flag |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 82 | safe | 1 | 24 | 0.08904639012225485 | 0.06511607764333781 | 0.7312601617419606 | 1 | 1.0 | 2.9524695834747856 | True |
| 86 | unsafe | 9 | 21 | 0.07560948358827242 | 0.0 | 0.0 | 29 | 0.72 | -6.995482992779127 | False |

Every unsafe endpoint raw proposal/projection row (shape[1,2,2,2], EV then gamma):

- s_prev: `[[[[0.6120189428329468, 0.597707986831665], [0.6015026569366455, 0.5891266465187073]], [[-0.6082435846328735, -0.5909212231636047], [-0.5973237156867981, -0.5872330069541931]]]]`
- p: `[[[[0.6414757966995239, 0.6238405704498291], [0.628511905670166, 0.6132559180259705]], [[-0.6373234987258911, -0.616515040397644], [-0.6242724657058716, -0.6122836470603943]]]]`
- s_end: `[[[[0.6414757966995239, 0.6238405704498291], [0.628511905670166, 0.6132559180259705]], [[-0.6373234987258911, -0.616515040397644], [-0.6242724657058716, -0.6122836470603943]]]]`
- previous_state_hash: `"e146b740cefd7a68f5a6987205fb8a404cf0b7c3c7d4eddcb56be7e7f0715252"`
- pre_box_hash: `"addaf2baa97d37463d674a94abe7e1c81f90dfa9b2aa2f6485e7e58cd78c9462"`
- state_hash: `"addaf2baa97d37463d674a94abe7e1c81f90dfa9b2aa2f6485e7e58cd78c9462"`
- lower_hash: `"3038871598507c69498016823833d29c08c25b2ba2ef5e0300838a0e7c7ab607"`
- upper_hash: `"45b5872f5534f792dac8c9939fcd1fb667f411f73568e40c0a89626956ec7ce7"`

Index86 has a nonzero proposed displacement (0.07560948358827242) but p equals s_end; thus zero pressure is not the zero-proposal fallback. Descending rank29 is shared with tied zero scores (28 strictly larger); percentile.72. Rank=1+strictly-greater count; percentile=fraction<=score. Full100 endpoint rows/ranks retained.

Reuse/verification: T068-D endpoint/input/freeze/label orchestration; accepted CommonBox construction and call. Verifier reconstructs the actual CommonBox from trace active/winner, matches recorded lower/upper bounds, applies it to copies of pre_box[k-1] on GPU, and requires exact EV/gamma equality to state[k]. Independent score uses coordinate-wise math.fsum rather than NumPy norm. Re-renders fixed endpoints on A6000 GPU1 and checks output hashes. Exactly 200 projection checks and 200 render checks; score max error 0.0, T99 error 0.0. All labels/flags/diagnosis agree. optimizer_runs=0; model_fits=0.

Tests: core3 passed4.47s; local affected9 passed15.32s; imports PASS; remote9 passed1.57s. Commands: `python -B -m pytest research_log/T069A research_log/T068D research_log/T067B -q`; `python -B -m research_log.T069A.run --out /media/wenchang/F/wjq/TTIE/runs/T069A-projection-pressure`; same with `research_log.T069A.verify`.

Run `20260921-115710-ttie-t069a-projection-pressure`, exit0 at2026-09-21T11:57:28+08:00, primary 1.9419454720045906 seconds. Release `/media/wenchang/F/wjq/TTIE/releases/20260921-ttie-t069a-projection-pressure`; output `/media/wenchang/F/wjq/TTIE/runs/T069A-projection-pressure`. Existing Python3.12/Torch2.4+cu121, physical GPU1, TF32off, sequentialMKL, OMP/MKL/OpenBLAS1. Primary eight-coordinate algebra CPU; verification/rendering GPU. 302 source/input bindings; separate label binding post-freeze.

Failures/deviations: none. No new quality metrics for hypothetical guarded output, selector, tuning/window/cumulative variant, fresh/final dataset access.

Raw archive `/media/wenchang/F/wjq/TTIE/shared/t069a/T069A_raw.tar`, 1341440 bytes, SHA256 `6b25ffd1d9e2b713a629ae0acd0eb49f118c8fa6d91e4e8fbc79feb443f37b96`. Recovery `/home/wenchang/asdasdsad/wjq/TTIE/shared/t069a/T069A_recovery.tar.gz`, F backup `/media/wenchang/F/wjq/TTIE/shared/t069a/T069A_recovery.tar.gz`, 3189978 bytes, SHA256 `e520c63155d1544f78d9ee11cb8fddc3ed569f3e37841b44823674407db24e04`; downloaded/hash-verified locally.

Conclusion/next step: exact final-transition projection pressure fails to identify the residual unsafe endpoint. Stop at SIGNAL_ABSENT and await the research lead. This closes only the authorized exact statistic; no alternative has been run. Exposed-cohort diagnosis, not fresh qualification.
