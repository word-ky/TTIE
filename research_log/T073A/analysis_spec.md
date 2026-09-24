# T073-A expanded UHD-LL main-table preregistration

This is an outcome-independent extension of the accepted T072-L three-method specification, not an edit to it. Target: exact canonical T072-I UHD-LL 150-image, native 3840×2160 RGB low-only cohort. The scientific question is whether frozen episodic Ours-TTT transfers to unseen low-light degradation relative to fixed, domain-adapted and zero-shot alternatives; it is not an in-domain SOTA claim. No UHD-LL clean/reference has been opened for this specification. No model execution or target metric belongs to T073-A.

## Row and provenance contract

`method_registry.json` fixes ten immutable method IDs and their current evidence status. Tier 1: RetinexFormer, SNR-Aware, PromptIR, PromptIR+DCTTA, MR. Illuminate, QuadPrior, Ours-Step0, Ours-TTT, and preferred ZERO-IG. GM-MoE is Tier 2 only and must not delay Tier-1 completion. The first two rows alone have accepted complete 150/150 output evidence, bound to T072-AZ-B manifest SHA256 `73c8d304c8bb88c4612a13598b5d4dd76378e0c3107bd97011941037a4427001`. Their source is the accepted LOL-v2 Real setting. Do not rerun either after reference outcomes. All other rows are `PENDING_OUTPUTS`, even when a source model or code exists.

Before a pending row can execute, its task-specific protocol must pin the exact official/source checkpoint(s), config and source-code hashes, source training dataset or official pretrained setting, native UHD-LL image I/O, target-low access, adaptation ordering/seed/hyperparameters where applicable, reset policy, and output hash/verification procedure. This preregistration deliberately does not claim checkpoint hashes for artifacts not yet obtained. If a declared protocol cannot run, record `UNRUN/BLOCKED` with reason; do not replace it with a target-tuned variant. PromptIR and DCTTA share the official three-task PromptIR base (`model.ckpt`), not a LOL-v2 Real checkpoint; any base-versus-adapted gain requires exact same source initialization. QuadPrior's published training is normal-light COCO, not paired low-light data. The MR. Illuminate author paper fixes a frozen Stable Diffusion UNet with QuadPrior VAE decoder and 25-step prompt-free DDIM path, but the exact pretrained weight revision must still be bound before execution. ZERO-IG's public repository cautions that supplied weights are not the paper-result weights, so the only acceptable candidate is a prospectively fixed, low-only official per-image recipe if reproducible. Do not silently substitute released weights and imply paper equivalence.

The planned table reports for **every method**: immutable ID/display name; precise source dataset/pretraining and checkpoint/config/source-code hashes (or explicit `PENDING_BINDING` before execution); paradigm; target-low access count/scope; adaptation granularity and reset; target GT during adaptation (`No`); mean/median PSNR; mean RGB-SSIM; Ours-TTT-minus-comparator paired PSNR mean, median, strict-positive win fraction and 95% CI where defined; and adaptation/inference time, peak VRAM, and test-time updated parameter/state size. The paired base→adapted gains are PromptIR+DCTTA minus PromptIR and Ours-TTT minus Ours-Step0. Mark an intrinsically noncomparable field `N/A` plus reason; do not invent a surrogate. Empty outcome cells remain `TBD` until the reference gate opens.

| Method ID | Current status | Source regime | Adaptation | GT in adaptation | Mean PSNR | Median PSNR | Mean RGB-SSIM | Time / VRAM / updated state |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| retinexformer | FROZEN_OUTPUTS | LOL-v2 Real | none | No | TBD | TBD | TBD | receipt / N/A updated state |
| snr_aware | FROZEN_OUTPUTS | LOL-v2 Real | none | No | TBD | TBD | TBD | receipt / N/A updated state |
| promptir | PENDING_OUTPUTS | official three-task base | none | No | TBD | TBD | TBD | TBD / N/A updated state |
| promptir_dctta | PENDING_OUTPUTS | same PromptIR base | domain-level | No | TBD | TBD | TBD | TBD |
| mr_illuminate | PENDING_OUTPUTS | official zero-shot diffusion prior | none | No | TBD | TBD | TBD | TBD / N/A updated state |
| quadprior | PENDING_OUTPUTS | normal-light COCO | none | No | TBD | TBD | TBD | TBD / N/A updated state |
| ours_step0 | PENDING_OUTPUTS | T070-A frozen source | none | No | TBD | TBD | TBD | TBD / N/A updated state |
| ours_ttt | PENDING_OUTPUTS | T070-A frozen source | episodic per image | No | TBD | TBD | TBD | TBD |
| zero_ig | PENDING_OUTPUTS | official per-image zero-shot recipe | episodic per image | No | TBD | TBD | TBD | TBD |
| gm_moe | PENDING_OUTPUTS, Tier 2 | official source setting to bind | none | No | TBD | TBD | TBD | TBD / N/A updated state |

## Metric and inference contract

`metric_plan.json` pins the accepted T072-L metric provenance without modification: T071-B `evaluate.py` calls T071-A `core.metrics`, with full-RGB PSNR and `ttie.ssim_transfer.rgb_ssim`; exact source SHA256 values and conversion details are in the machine-readable plan. Neither the older seed-7 bootstrap helper nor an alternative SSIM implementation is substituted. No crop, resize, output quantization or brightness matching is allowed. Future metric evaluation uses all 150 images for every included row, exact canonical image pairing and native geometry; any missing/duplicate/nonfinite/geometry/binding/reference-mapping error aborts the whole comparison, without exclusion or imputation.

Per method report mean PSNR, median PSNR and mean RGB-SSIM. For every scientifically defined comparison, calculate per-image paired differences before aggregation. Headline sign: `Ours-TTT − comparator`. Adaptation-gain sign: `adapted − base`. Report mean and median paired PSNR/RGB-SSIM differences and win fraction; strict `delta > 0` wins, ties do not, denominator 150. For each paired mean delta, use the **same** T072-L PCG64 seed `20260922` matrix of 10,000 replacement resamples of 150 canonical image indices, in canonical order. The 95% percentile CI uses 0.025/0.975 linear-interpolated quantiles. Report effect sizes and CIs, not unsupported significance or superiority thresholds. No subgroup, outcome-driven exclusion, or post-result statistic change is registered.

## Information boundary and release gate

UHD-LL references stay sealed until **all Tier-1 rows declared for the final table** have complete native target-low outputs frozen, hashed and independently verified; inclusion of ZERO-IG and GM-MoE must be locked before opening references. Their status cannot change based on future PSNR/SSIM, baseline outcomes, or qualitative target-reference inspection. No such signal may choose source checkpoint, adaptation hyperparameters, stopping, retry, sample inclusion, or execution rescue. If a Tier-1 method proves unreproducible, surface `UNRUN/BLOCKED` and obtain an explicit prospective research-lead table-scope decision before reference opening; do not silently drop it. LSRW and Target-3 remain separate later targets.

## Source references and limits

- Historical T072-L specification: immutable commit `63b8f465e15b164eb5262f80d37a64ab1ce50056`, spec SHA256 `df9c5e4a6c2c5ee8cf812c4535938ecb80d5f6b34fbc659d2fc8302923f93391`.
- Accepted T072-AZ-B complete-output manifest: immutable commit `c45d9bc4f821d2e6fdf0ccff86ee7cb77492444b`, SHA256 `73c8d304c8bb88c4612a13598b5d4dd76378e0c3107bd97011941037a4427001`.
- Official public-source descriptions: [DCTTA](https://github.com/tonia86/DCTTA), [QuadPrior](https://github.com/daooshee/QuadPrior), [ZERO-IG](https://github.com/Doyle59217/ZeroIG), [MR. Illuminate author paper](https://arxiv.org/html/2412.13401). These identify intended regimes, not yet local executable binding hashes.

No outcome numbers are asserted here. The ccf-experiment-designer evidence-first template guided the row/evidence matrix and kept all result cells explicitly `TBD`.
