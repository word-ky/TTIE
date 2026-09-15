# T043-A — DONE: matched-gain early-state quality bridge not supported / mixed

**REFERENCE_EVALUATION_ONLY.** At the exact matched gain1.75, T042 step10-legacy outputs are worse on average than T041 selected-legacy outputs: paired **ΔPSNR=-1.864979122440 dB**, **ΔRGB-SSIM=-0.036338683125**. The sole gate requires meanΔPSNR>=+0.50dB AND meanΔRGB-SSIM>=0; neither condition passes. Verdict: **matched-gain early-state quality bridge not supported / mixed**. Every one of the100 matched pairs is retained; no thresholds or per-image choices were adjusted.

T042's restored local gradient alignment does not establish better quality at the substituted state. The original selected state can have lower reference error while its learned local direction is less restorative. This result blocks the proposed quality bridge for this fixed substitution; it does not negate T042's local-field measurement. No controller, global step10 checkpoint or inference change is justified by this audit alone.

## Frozen inputs and information boundary

Executed/tested source **bacb3dbc7f1f465d84fb8c2ff436fff3678441f7**, branch `codex/T043A-matched-gain-quality`, PR [#68](https://github.com/word-ky/TTIE/pull/68), base12b89ca1 after accepted T042 mergec03e1642b225575fa6c28dd4c8bf77333c348d7c. Evidence/main-mailbox SHAs are in final delivery. The accepted `ttie/ssim_transfer.py` implementation is reused unchanged; its source hash and every evaluation/test/helper/cohort binding are recorded in output_binding.json.

Exact T036100 cohort SHA **279c74b335999d6631d436c79ea80e9e2cccfc4b97cb7c0b992829d04cfaf40b**. Baseline is the retained T041 selected-legacy+gain1.75 output at fixed index1 in Stage-A run20260915-122826; candidate is retained T042 step10-legacy+gain1.75 at index0 in run20260915-133013. These100 normal pairs were already reference-used by T041/T042; no fresh cohort or official-test image is accessed.

- T041 accepted freeze SHA **b2612273d37eb982c40736d5d3434cfa717666a8cad70e18574f41968b08d64b**.
- T042 accepted freeze SHA **a5b0799ee370bfd482393fce59895d904008cee9c577d85b385c3ce19dd92c36**.

Both accepted per-image tensor-file hashes, names/order, output indices, native shapes, raw-state hashes and output hashes are verified before reference access. Active gain raw values equal the accepted1.75 conversion exactly and inactive gain is identity. Both gates/assets are equal; candidate legacy-step metadata is10 and the raw legacy hash matches its frozen provenance, while baseline selected-step metadata matches T041. All200 native output tensors are loaded directly and cloned only for read-only scoring; **no rendering, adaptation or selection is run**.

All100 matched pair identities were persisted in output_binding.json at **2026-09-15T05:59:33.693239+00:00**, SHA **10c26674fba293304b6b75b7fe517a561e53e7ba0d59d0ae013eb779d05dbfb9**, with **normal opens0**. Image.open is prohibited during all output binding. Only afterward is a normal-only allowlist enabled; first normal decode **2026-09-15T05:59:33.700949+00:00**. Each normal hash is checked against the accepted cohort before decode. All200 cached output hashes and original tensor-file hashes are checked unchanged after scoring and again at archive. Zero optimizer updates, state changes, selection changes, per-image output choices or test access.

## Exact evaluation convention and results

Use the accepted T026/T036 native400x600 RGB convention: outputs are retained float32[0,1] tensors, cast to float64; normal RGB uint8 is divided by255 in float64, rounded to float32 then cast to float64. No resizing, crop, luminance-only conversion, reference brightness matching or normalization. Per-image PSNR is -10log10 of full RGB mean squared error. RGB-SSIM uses an11x11 Gaussian window sigma1.5, population covariance, K1=.01/K2=.03, data_range1, half-sample symmetric reflect padding, all image borders retained and mean over RGB; all arithmetic float64.

| Metric | Baseline mean | Candidate mean | Paired mean | Median | p10 | p90 | Positive / negative / zero |
|---|---:|---:|---:|---:|---:|---:|---:|
| psnr | 12.044154697929 | 10.179175575489 | -1.864979122440 | -1.173922142053 | -5.288450143944 | 0.906931018194 | 15 / 85 / 0 |
| ssim | 0.349457464353 | 0.313118781228 | -0.036338683125 | -0.032326338482 | -0.127321556471 | 0.018438951760 | 36 / 64 / 0 |

Quantiles are NumPy linear interpolation, paired means/medians use100 candidate-minus-baseline differences. Positive/negative/zero use exact sign, without a tolerance threshold. Tail cases are descriptive only; none is used to change outputs or the primary gate.

![Frozen matched-gain paired quality changes](T043A_result/quality_deltas.png)

| Metric | Tail | Image | Candidate-minus-baseline |
|---|---|---|---:|
| psnr | worst | Train/Low/low00549.png | -8.593399277706 |
| psnr | worst | Train/Low/low00232.png | -8.048091149440 |
| psnr | worst | Train/Low/low00600.png | -8.043206104970 |
| psnr | worst | Train/Low/low00070.png | -7.857080552062 |
| psnr | worst | Train/Low/low00111.png | -7.699568187821 |
| psnr | best | Train/Low/low00559.png | 7.117374976282 |
| psnr | best | Train/Low/low00644.png | 6.553721056260 |
| psnr | best | Train/Low/low00348.png | 5.138006745139 |
| psnr | best | Train/Low/low00198.png | 4.991542474008 |
| psnr | best | Train/Low/low00271.png | 4.645428282154 |
| ssim | worst | Train/Low/low00670.png | -0.177229361267 |
| ssim | worst | Train/Low/low00671.png | -0.175658746459 |
| ssim | worst | Train/Low/low00018.png | -0.161250978858 |
| ssim | worst | Train/Low/low00063.png | -0.153102433661 |
| ssim | worst | Train/Low/low00472.png | -0.152090570358 |
| ssim | best | Train/Low/low00644.png | 0.165609825722 |
| ssim | best | Train/Low/low00559.png | 0.130034720673 |
| ssim | best | Train/Low/low00198.png | 0.097615037167 |
| ssim | best | Train/Low/low00039.png | 0.092189813452 |
| ssim | best | Train/Low/low00145.png | 0.075567692459 |

The largest PSNR recovery is low00559 (+7.117375dB), but the aggregate and median are negative and85/100 pairs lose PSNR. This is not a globally beneficial early-state substitution. No original variable-gain T036 state is scored or substituted into the matched comparison.

## Independent verification and execution

Run **20260915-135922-ttie-t043a-quality**, release20260915-135847-ttie-t043a-quality, completed exit0 on the A6000 server. This is CPU evaluation using the unchanged accepted SciPy metric, with no model or adaptation work. Main scoring including binding took **17.423726s**; independent replay **15.713936s**. Baseline SSIM cross-check against the full independent skimage SSIM map passed1test1.44s; T0432local tests passed0.17s and2server tests0.09s.

Independent replay separately reloads both frozen tensors and the same normal for every pair, checks tensor-file/output/normal hashes, recomputes PSNR via torch float64 reduction/log10 and RGB-SSIM via explicit11-tap separable scipy.convolve1d weights, without calling the main T043 metric/aggregation/verdict helpers. It recomputes all paired deltas, population means, means/medians/p10/p90/sign counts, best/worst cases and the single gate using standard-library statistics. **100 pairs /400 output-metric evaluations /638 checked scalars**, max absolute metric/aggregate discrepancy **7.105427357601002e-15 <=1e-10**, PASS. The independent replay itself verifies the pre-reference binding timestamp. Source/checkpoint selection is never recomputed.

## Failures, artifacts and next step

No failures, restarts, deviations or scientific changes; one main evaluation and one independent replay. All metrics finite. No GPU metric replacement was introduced because exact reuse of the accepted evaluation implementation was the requirement. No optimizer, renderer, CLIP/energy call, policy, training, fresh cohort or official-test access.

Evidence archive **44615 bytes**, SHA **ba50381b7a511216943ae5ef56ccbc1bc619245ad9fd2da76ceecbe8695162ba**, verified under home/F shared/t043a and locally. It contains pair-output binding, main/independent receipts, per-image metrics, summary and run logs. Original immutable tensor bundles remain at their bound T041/T042 Stage-A paths, with existing F backups `shared/t041a/T041A_full.tar` and `shared/t042a/T042A_full.tar`; their200 original tensor files were reverified unchanged. Final Git-exact source/evidence/mailbox recovery and reports/plots are mirrored under project research_log and both server roots. Exact evaluation/replay commands/environment are in run/run.sh.

Stop after this evaluation bridge. Recommend research-lead reassessment of the distinction between current-state restoration error and the learned direction before choosing an intervention. Do not implement a stopping/freezing controller from T042 alone, rerun adaptation, or seek a different step/gain in this cycle.

matched-gain early-state quality bridge not supported / mixed
