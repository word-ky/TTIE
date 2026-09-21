# T068-C — DONE / TAIL_INEFFICIENCY_SIGNAL_ABSENT

The fixed three-transition endpoint inefficiency score misses the only unsafe T067-C selected endpoint: unsafe above threshold0/1, safe above threshold2/99. This is a diagnosis only; all original T067-C choices remain unchanged.

Source: `dc9f766c16ad25971bb1a36a52eedc0a035733c2`; branch: `codex/T068C-tail-inefficiency`; authorization: `cf4075ea311bf5b7e845937b9627758aa3272b84`.

Exact statistic: k0=max(k_FS,k_lambda-3); p_tail=clip((L[k0]-L[k_lambda])/max(D_total,1e-12),0,1); q_tail=M_tail/max(M_total,1e-12); R=log((q_tail+1e-12)/(p_tail+1e-12)). Float64 RMS rendered RGB motion; stated floors and singleton override only. Frozen lambda=.875, rho=.9857470621423519, probability threshold.5, model/renderer/trajectory unchanged. No optimizer rerun/model fit.

T99=0.6324473434957039 is exactly sorted development R[98], using100 target-free development scores and no development reference quality. Development rows above T99=1. Development freeze UTC `2026-09-21T00:43:00.611501+00:00`, SHA256 `7774d29d40cc16d292a907a9928b15df8b9291beb06ce4425e93a0b2e850939a`.

All100 transfer endpoint scores/component values/input-state-output/model-rule hashes, strict R>T99 flags and descriptive ranks were frozen at `2026-09-21T00:43:12.484889+00:00`, SHA256 `3952807057fc0ac2e163c084a4c2b913b1d2be56629ef153aa8853601cd2a434`, reference_reads=0. First transfer quality-read boundary UTC `2026-09-21T00:43:12.490411+00:00`; only then check evaluation hashes and join existing T067-C endpoint labels.

| Group | Endpoints | R>T99 |
|---|---:|---:|
| Unsafe | 1 | 0 |
| Safe | 99 | 2 |

Unsafe and above-threshold endpoint rows:

| Index | Safety | k_FS | k_lambda | k0 | p_tail | q_tail | R | Descending rank | Empirical percentile | Margin vs T026 | Flag |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | safe | 8 | 19 | 16 | 0.06422026461100645 | 0.1863027878665453 | 1.0650544328430303 | 1 | 1.0 | -4.543491676163251 | True |
| 1 | safe | 0 | 9 | 6 | 0.15825236740410759 | 0.31182694380906756 | 0.6782573458309783 | 2 | 0.99 | 3.514738419272984 | True |
| 86 | unsafe | 9 | 21 | 18 | 0.1572910111093997 | 0.2504241042516345 | 0.4650582340311617 | 15 | 0.86 | -6.995482992779127 | False |

Rank1 is the largest R; ties share rank1+number strictly greater. Empirical percentile is fraction<=R. Complete score/rank/component rows for all99safe and1unsafe endpoints are retained in joined_endpoints.json; no label-derived threshold or ranking change.

Reuse/validation: accepted T068-B target-free development motions, T066-A input/model/renderer and transfer history, T067-C endpoint freeze. Primary transfer motion is GPU float64; verifier reconstructs both cohorts/endpoints, rerenders 4015 interval states on A6000 GPU1, independently computes CPU float64 dot RMS, score/T99 and exact flags/diagnosis. Primary T067-C labels are independently crosschecked against accepted T066-B per-state labels at the same endpoint. No new reference image scoring is needed or performed.

Independent verifier PASS: score/component maximum discrepancy `6.294964549624638e-14`; independent T99 `0.6324473434957248` (error `2.098321516541546e-14`); all flags and diagnosis agree. optimizer_runs=0, model_fits=0.

Tests: core3 passed1.57s; local final10passed1skipped15.49s, imports PASS; remote11passed1.98s including GPU RMS. Commands: `python -B -m pytest research_log/T068C research_log/T068B research_log/T067B -q`; `python -B -m research_log.T068C.run --out /media/wenchang/F/wjq/TTIE/runs/T068C-tail-inefficiency`; same with `research_log.T068C.verify`.

Run `20260921-084251-ttie-t068c-tail-inefficiency`, exit0 at2026-09-21T08:43:48+08:00, primary 12.692659895983525 seconds. Release `/media/wenchang/F/wjq/TTIE/releases/20260921-ttie-t068c-tail-inefficiency`; output `/media/wenchang/F/wjq/TTIE/runs/T068C-tail-inefficiency`. Existing Python3.12/Torch2.4+cu121, CUDA_VISIBLE_DEVICES=1, TF32 off, sequential MKL, OMP/MKL/OpenBLAS1. 290 bound source/input files; separate two-file evaluation binding used post-freeze.

Failures/deviations: none. No hypothetical guarded-output PSNR/SSIM, stopping or rollback selector, alternative statistic, tuning, fresh cohort or final-set access.

Raw: `/media/wenchang/F/wjq/TTIE/shared/t068c/T068C_raw.tar`, 1034240 bytes, SHA256 `fb7177129d8f24faaafc5c1f0d101a2224d494faeeb1d0bc0bcb8b55664ce974`. Recovery: `/home/wenchang/asdasdsad/wjq/TTIE/shared/t068c/T068C_recovery.tar.gz` with F backup `/media/wenchang/F/wjq/TTIE/shared/t068c/T068C_recovery.tar.gz`, 3171151 bytes, SHA256 `db7797e30352d5a4f5c83753fe2f7326b2302c255ac43fd2a7523aa50527efd9`; downloaded and hash-verified locally.

Conclusion/next step: the exact statistic does not make the residual unsafe endpoint an extreme outlier under the target-free development threshold. Report SIGNAL_ABSENT and stop; await the research lead. No deployable guard is justified by this diagnostic, and this exposed-cohort result is not fresh qualification.
