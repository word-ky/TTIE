# T024-A: frozen LOL-v2 Real baseline protocol

**Conclusion: baseline coverage insufficient.** All five requested families were audited; no model was run, no official test image was decoded, and no Ours code/weights/settings changed. This completes the protocol task, not the benchmark.

The strict matched-data main group currently has two provenance-eligible supervised methods and no eligible 2025 release. Zero-DCE++ has genuine official target-free code/weights but uses external SICE training; its separate-stratum interpretation cannot repair the missing 2025 coverage. Readiness is not claimed.

| Baseline | Future designated mode | Official-test class | TTIE validation class |
|---|---|---|---|
| Retinexformer | default_no_gt_mean | FINAL_TEST_READY | VALIDATION_RETRAIN_REQUIRED |
| SG-LLIE | lolv2_paper_variant | UNSUPPORTED | UNSUPPORTED |
| SNR-Aware Low-Light Enhancement | ttie_native_pad16 | FINAL_TEST_READY | VALIDATION_RETRAIN_REQUIRED |
| LLFormer | lolv2_matched_unavailable | UNSUPPORTED | UNSUPPORTED |
| Zero-DCE++ | native_scale1_external_sice | UNSUPPORTED | UNSUPPORTED |

FINAL_TEST_READY denotes official checkpoint/data and target-free forward eligibility; it does not mean a TTIE exporter was implemented, checkpoint binary hashed, historical test-blind selection proven, or a reproduction completed. UNSUPPORTED is scoped to matched LOL-v2 use; it does not imply a model has no release. Exactly one target-free mode is designated per family; unsupported modes remain excluded from the main table.

## Retinexformer

[Official repository](https://github.com/caiyuanhao1998/Retinexformer) at `1e9a0efce4b306b6701b824768370ff26066c32a` (2026-05-23T16:09:21Z). [Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Cai_Retinexformer_One-stage_Retinex-based_Transformer_for_Low-light_Image_Enhancement_ICCV_2023_paper.pdf): Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement, ICCV 2023.

**License:** MIT (LICENSE.txt).

**Checkpoint:** `pretrained_weights/LOL_v2_real.pth`; [official locator](https://drive.google.com/drive/folders/1ynK5hfQachzc8y96ZumhkPPDXzHJwaQV?usp=drive_link). Official README identifies dataset checkpoint; binary not downloaded or hashed in this task. Binary SHA256 remains unbound; this audit did not download/load model weights.

**Training:** LOL-v2 Real official Train (689 paired images); all 100 TTIE validation pairs overlap Current YAML: 256 crops, batch 8, 150000 iterations, Adam 2e-4, L1, mixup. Paper: 128 crops, batch 8, 250000 iterations. Configuration: `Options/RetinexFormer_LOL_v2_real.yml`. YAML validation paths point to official Test. Historical checkpoint selection and exact training log are not independently bound; do not claim legacy test-blind model selection.

**Image path:** RGB float32 /255; reflect-pad right/bottom to multiples of 4, unpad. Native 400x600 needs no padding. Default self-ensemble disabled; large-image split branch irrelevant here. Clamp [0,1]; original PNG export quantizes to uint8.

**Target-access audit:** Dataset test entrypoint loads target before forward for scoring. With GT_mean absent/false, target does not feed the enhancement network or output processing. --GT_mean rescales output by target-gray mean / output-gray mean and clips, hence rejected. Future low-only exporter must avoid target loading entirely until all outputs are frozen.

**Repository scoring:** Default script PSNR from floating RGB MSE; SSIM uses uint8 images, Gaussian 11/sigma1.5, population covariance and valid interior [5:-5], averaged over RGB. Not identical to TTIE full-map reflect SSIM. Saved PNG is 8-bit.

**Modes:**
- `default_no_gt_mean`: final `FINAL_TEST_READY`; validation `VALIDATION_RETRAIN_REQUIRED`. Official LOL-v2 checkpoint, GT_mean=false, self_ensemble=false; future TTIE low-only float exporter, same native geometry.
- `gt_mean`: final `REJECT_TARGET_ASSISTED`; validation `REJECT_TARGET_ASSISTED`. Reference mean changes enhancement; excluded even when reported as default paper comparison
- `self_ensemble_no_gt_mean`: final `FINAL_TEST_READY`; validation `VALIDATION_RETRAIN_REQUIRED`. Optional target-free eight-way ensemble, not selected for main

**Published LOL-v2 context, never reproduced results:** 22.80 dB / 0.840, Original ICCV paper Table 1; Original paper context; not a reproduced TTIE uniform-metric result; 27.71 dB / 0.856, Inspected official README table; Repository GT-mean convention; REJECT_TARGET_ASSISTED, non-comparable

Eligibility label is provenance/inference eligibility, not a completed reproduction. Exact binary hash and low-only export remain future work.
Current YAML versus paper iteration/crop mismatch and legacy official-Test validation must remain visible.

Source anchors and byte hashes are listed in the machine-readable manifest and evidence index.

## SG-LLIE

[Official repository](https://github.com/minyan8/imagine) at `0b96263167b7511bede010537b64f8b44ba240a4` (2025-04-23T04:51:25Z). [Paper](https://arxiv.org/html/2504.14075v1): Towards Scale-Aware Low-Light Enhancement via Structure-Guided Transformer Design, CVPR Workshops / NTIRE 2025.

**License:** No project-level license found at inspected commit; vendored package licenses do not license the project.

**Checkpoint:** `Enhancement/weights/model.pth`; [official locator](https://mcmasteru365-my.sharepoint.com/:f:/g/personal/dongw22_mcmaster_ca/Em4rtdZsS3NKtE2K-pTXCXsBSrwmB_gPwXtd0eldBUn6Ig?e=pAZVvC). NTIRE checkpoint exists as a Git blob and README external link. No official LOL-v2 Real checkpoint binding found.. Binary SHA256 remains unbound; this audit did not download/load model weights.

**Training:** Released configuration/checkpoint: NTIRE 2025. Paper also reports LOL-v2 Real; release does not bind that run. Paper LOL recipe: 384 patches, batch 4, adjustment layer omitted. Released YAML is NTIRE 1600-patch configuration, not LOL. Configuration: `Enhancement/Options/Ntire25_LowLight.yml`. Exact LOL-v2 training log/checkpoint selection unverified; cannot substitute NTIRE checkpoint as matched LOL-v2 training.

**Image path:** Released test.py: input RGB /255 plus structure prior computed from low image only; reflect-pad both to multiple 32, eight-way self-ensemble, unpad, clamp [0,1], uint8 PNG. Actual test imports UHDM_arch.py. No LOL-specific runnable recipe bound.

**Target-access audit:** Released NTIRE forward uses low image and its prior only, no normal target or PSNR feedback. Prior normalization is input-derived, not target-mean matching. Paper adjustment module is a learned train-time mechanism; no basis to label it test-target-assisted. Released LOL variant remains unverified.

**Repository scoring:** Released test.py saves images without scoring. Generic BasicSR metric functions exist but do not establish exact LOL paper quantization, border or scoring settings.

**Modes:**
- `lolv2_paper_variant`: final `UNSUPPORTED`; validation `UNSUPPORTED`. Designated target-free LOL variant would omit adjustment per paper; no official matched checkpoint/config bound.
- `ntire_released`: final `UNSUPPORTED`; validation `UNSUPPORTED`. Target-free NTIRE inference exists; not a matched LOL-v2 checkpoint/recipe.

**Published LOL-v2 context, never reproduced results:** 22.84 dB / 0.859, Official paper Table 2; Published LOL-v2 Real context only; exact released LOL metric/inference binding unavailable

Missing 2025 method blocks readiness independently of all other choices.
Do not claim all official weights are absent: the 51 MB NTIRE Git checkpoint exists.

Source anchors and byte hashes are listed in the machine-readable manifest and evidence index.

## SNR-Aware Low-Light Enhancement

[Official repository](https://github.com/JIA-Lab-research/SNR-Aware-Low-Light-Enhance) at `1113144c82adc8bcc4a9ec27749ed75f196a4e4d` (2022-11-21T06:45:49Z). [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_SNR-Aware_Low-Light_Image_Enhancement_CVPR_2022_paper.pdf): SNR-Aware Low-Light Image Enhancement, CVPR 2022.

**License:** No project-level license file or grant found at inspected commit.

**Checkpoint:** `LOLv2_real.pth`; [official locator](https://drive.google.com/file/d/1g3NKmhz7WFLCm3t9qitqJqb_J7V4nzdb/view?usp=sharing). Official README names LOLv2_real.pth in pretrained archive; binary not downloaded or hashed. Binary SHA256 remains unbound; this audit did not download/load model weights.

**Training:** LOL-v2 Real official Train 689; overlaps frozen TTIE validation Training video_base3, test video_base4_m; low_light_transformer nf64. YAML batch4, 600000 iterations, Adam4e-4, Charbonnier. Dataset training path resizes to 608x400 despite GT_size128 config field. Configuration: `options/train/LOLv2_real.yml; options/test/LOLv2_real.yml`. Train config validation path points to Train; exact distributed checkpoint training/selection log not independently verified.

**Image path:** Test dataset loads native RGB float and 5x5 blurred-low features. Official test4 bilinearly resizes image/SNR mask to H400 W608, restores output to original dimensions. TTIE main instead predeclares reflection padding to multiple16, native direct test()/network forward, then unpad and clamp; this exporter is planned, not implemented or tested. Two stride2 layers plus 4x4 feature patches motivate multiple16.

**Target-access audit:** feed_data stores GT by default and script loads targets for metrics, but test()/test4 network and SNR mask use only low and blurred low. Future exporter uses need_GT=False and avoids target visuals. No target-mean rescaling found.

**Repository scoring:** tensor2img clamps and rounds to uint8, with RGB/BGR conversion. test_LOLv1_v2_real.py PSNR scores uint8; SSIM calculation is commented and variable set to 0, so script SSIM is a placeholder, not a valid reproduced SSIM. Generic util Gaussian-valid SSIM is not called there.

**Modes:**
- `ttie_native_pad16`: final `FINAL_TEST_READY`; validation `VALIDATION_RETRAIN_REQUIRED`. Official checkpoint with predeclared target-free native pad/unpad exporter; adapter not yet implemented. Report as protocol adaptation, not exact test4 reproduction.
- `official_test4_resize`: final `FINAL_TEST_READY`; validation `VALIDATION_RETRAIN_REQUIRED`. Official target-free forward but resize400x608/back; preserve for provenance, exclude from native-input main.
- `official_direct_test`: final `FINAL_TEST_READY`; validation `VALIDATION_RETRAIN_REQUIRED`. Official direct forward; fixed native padding needed for patch geometry, no metric-based mode selection.

**Published LOL-v2 context, never reproduced results:** 21.48 dB / 0.849, Original CVPR paper LOL-v2 Real table; Paper context; official inspected script does not reproduce its SSIM. Not TTIE measured values.

Original official dvlab-research URL redirects to JIA-Lab-research canonical repository.
Native exporter needs a later non-test smoke run; no claim of bitwise equivalence to test4 resize mode.

Source anchors and byte hashes are listed in the machine-readable manifest and evidence index.

## LLFormer

[Official repository](https://github.com/TaoWangzj/LLFormer) at `ed5e7bf61d2dc64ba3b4552b90eb68a38f6c5302` (2023-11-30T02:22:00Z). [Paper](https://arxiv.org/abs/2212.11548): Ultra-High-Definition Low-Light Image Enhancement: A Benchmark and Transformer-Based Method, AAAI 2023.

**License:** CC BY-NC-SA 4.0; academic research only (LICENSE).

**Checkpoint:** `checkpoints/LOL/models/model_bestPSNR.pth`; [official locator](https://drive.google.com/drive/folders/1J7NvvPsCtT0j8Rd9ombJ6sVIC6v0Xweb?usp=share_link). Official LOL-v1 checkpoint; no LOL-v2 checkpoint/config found. Binary SHA256 remains unbound; this audit did not download/load model weights.

**Training:** Official LOL recipe is LOL-v1 485 train /15 test, plus separate UHD-LOL/FiveK releases; not LOL-v2 689. LOL training128 patches, batch8,4000 epochs, learning rate1e-4 to1e-6. Configuration: `configs/LOL/train/training_LOL.yaml`. LOL-v1 test used as validation; model_bestPSNR name denotes historical supervised validation selection, not inference target assistance.

**Image path:** PIL RGB tensor [0,1], reflect-pad to multiple16, network, clamp, unpad, uint8 output; target-free input-only test.py. Full-frame official LOL route, not UHD patch test.

**Target-access audit:** test.py uses no target/reference or quality metric to alter output. Separate evaluation.py scores saved files. Training validation checkpoint selection is distinct from test-time output selection.

**Repository scoring:** evaluation.py uses skimage PSNR and SSIM(multichannel=True) on RGB uint8; no Gaussian options, hence legacy default uniform7/sample covariance and version-sensitive API. Not TTIE Gaussian full-map float pipeline.

**Modes:**
- `lolv2_matched_unavailable`: final `UNSUPPORTED`; validation `UNSUPPORTED`. No official matched LOL-v2 training/checkpoint recipe.
- `lolv1_transfer`: final `UNSUPPORTED`; validation `UNSUPPORTED`. Official target-free LOL-v1 model exists; cross-dataset transfer would be separately labeled, not matched LOL-v2 main.

**Published LOL-v2 context, never reproduced results:** No own-paper LOL-v2 Real number bound; experimental cells remain null.

No LOL-v2 number found in original paper; do not copy later unofficial retraining tables.
LOL-v1 versus LOL-v2 scene overlap has not been verified.

Source anchors and byte hashes are listed in the machine-readable manifest and evidence index.

## Zero-DCE++

[Official repository](https://github.com/Li-Chongyi/Zero-DCE_extension) at `09f202b690f82da939b8e6ec8535960ae97ad8bd` (2022-06-19T14:34:54Z). [Paper](https://arxiv.org/abs/2103.00860): Learning to Enhance Low-Light Image via Zero-Reference Deep Curve Estimation, IEEE TPAMI (online 2021; print 2022) 2021.

**License:** CC BY-NC 4.0; academic research only (README).

**Checkpoint:** `Zero-DCE++/snapshots_Zero_DCE++/Epoch99.pth`; [official locator](https://github.com/Li-Chongyi/Zero-DCE_extension/blob/09f202b690f82da939b8e6ec8535960ae97ad8bd/Zero-DCE%2B%2B/snapshots_Zero_DCE%2B%2B/Epoch99.pth). Official checkpoint is committed as Git blob; binary not downloaded. Binary SHA256 remains unbound; this audit did not download/load model weights.

**Training:** External SICE Part1: paper 3022 images from360 exposure sequences,2422 train/600 validation; no paired normal targets used for optimization. Paper512x512; code training resize512, batch8,100 epochs, Adam1e-4, scale_factor1. Exact checkpoint sample manifest unavailable. Configuration: `Zero-DCE++/lowlight_train.py`. Epoch99 official artifact, no inference-time target or quality selection; SICE external data does not meet literal LOL-v2 official-train-only main stratum.

**Image path:** Default lowlight_test.py scale_factor12 crops H,W down to multiples12 (400x600 ->396x600), with bilinear low-res curve estimation. Future separately labeled external-data mode predeclares official supported scale_factor1, no resize/crop, native full frame. Eight shared-curve updates, RGB float input; export float before torchvision image quantization.

**Target-access audit:** Zero-reference train losses and low-only inference; no target image, target mean or PSNR output selection. Input-dependent curve prediction is admissible. External data eligibility issue is not target assistance.

**Repository scoring:** Official test script saves outputs without PSNR/SSIM. Original paper metrics are not a bound LOL-v2 Real uniform-metric recipe; no own-paper LOL-v2 numbers recorded.

**Modes:**
- `native_scale1_external_sice`: final `UNSUPPORTED`; validation `UNSUPPORTED`. Verifiable target-free external-SICE recipe; outside literal official-LOL-train-only main class. Research lead may authorize separately labeled external/zero-reference stratum later.
- `default_scale12_crop`: final `UNSUPPORTED`; validation `UNSUPPORTED`. Target-free official script but crops full-frame border and uses external SICE training; not main.

**Published LOL-v2 context, never reproduced results:** No own-paper LOL-v2 Real number bound; experimental cells remain null.

UNSUPPORTED here concerns requested matched-data use, not absence of official code/checkpoint.
If external-data zero-reference comparison is approved later, retain scale1 before any metrics. It still cannot fill missing 2025 coverage.

Source anchors and byte hashes are listed in the machine-readable manifest and evidence index.

## Frozen future main-table protocol

- **dataset:** LOL-v2 Real official100-pair test; untouched in this task
- **validation_split_sha256:** b88c8347005984b5523b117b52c0c068672fe172eb7c9aa5b60102d350e2d85b
- **selection:** Freeze Ours and all baseline modes/checkpoints using train/validation only before official-test decoding. T022-C remains candidate, not replaced by T023-A.
- **execution:** One official-test evaluation after configuration freeze; GPU preferred for model computation; no run authorized by this document alone.
- **reference_boundary:** Export/freeze all images, decisions and hashes for all admitted methods before loading any normal-light test reference. References score only; no reranking, mode/step/checkpoint selection, output normalization or retries selected by metrics.
- **input:** Native H400 W600 RGB float32 [0,1], no image resampling/crop. Architecture-required reflection padding then exact unpadding only; network internal operations retained.
- **output:** Native full-frame RGB float32 clipped [0,1], freeze float artifact before display PNG quantization. No gamma/brightness/mean matching or per-image output normalization.
- **psnr:** {"formula": "-10*log10(mean((output-reference)^2))", "domain": "float64 evaluation of frozen RGB [0,1], all pixels/channels", "perfect_match": "positive infinity", "aggregation": "arithmetic mean of per-image PSNR"}
- **ssim:** {"implementation": "ttie/ssim_transfer.py:rgb_ssim", "window": 11, "sigma": 1.5, "K1": 0.01, "K2": 0.03, "data_range": 1, "covariance": "population", "padding": "scipy.ndimage reflect, half-sample symmetric", "crop": 0, "channels": "mean over all RGB/full-frame map", "dtype": "float64", "aggregation": "arithmetic mean of per-image SSIM", "implementation_sha256": "01d227a8b4caaf8f5705d9c18a5ef685367b6ccf80588d212be92830a04f8556", "hash_basis": "Git blob at base_main"}
- **reporting:** Only actually computed uniform-pipeline results enter experimental table. Literature numbers in separate context field, never reproduced values.
- **validation:** All-689 supervised checkpoints require retraining on589 non-validation pairs before any fair validation comparison; no retraining authorized this cycle.
- **legacy_caveat:** Historical official checkpoint validation/test monitoring must be disclosed; FINAL_TEST_READY does not assert historically test-blind checkpoint selection.
- **runner_status:** Planned exporters only, no baseline code changed or executed

## Prioritized later work

1. Research-lead review missing2025 coverage and training-stratum policy; no sixth method or unofficial substitution in T024-A.
2. Retinexformer: bind downloaded checkpoint SHA256; implement default low-only float exporter; smoke-check using non-test data. Retrain589 only if validation comparison requested.
3. SNR: bind archive/checkpoint SHA256; implement predeclared native pad16 exporter and non-test geometry test; report departure from official resize mode.
4. Zero-DCE++: only after separate external-data stratum authorization, native scale1 low-only smoke check.
5. SG-LLIE/LLFormer: wait for verifiable matched official releases or research-lead scope decision. Do not synthesize missing checkpoint provenance.
6. Freeze final Ours and admitted recipes, then single target-free official-test run with scoring after every output is frozen.

## Audit evidence and limitations

`T024A_baseline_manifest.json` is the complete machine-readable protocol; CSV is a compact projection. `T024A_source_files.json` binds official raw source bytes; `T024A_repository_snapshots.json` binds repository commits and Git checkpoint blob identities. `T024A_evidence.md` provides line-addressed official code links. Raw source snapshots and four downloaded papers stay in the project-local `T024A_sources/` audit cache; they are not republished in this PR. No unofficial forks were used.

Official README weight locators are provenance evidence, not proof that external downloads are currently accessible or that bytes reproduce the paper. Missing exact training logs, publication/config differences, unspecified project licenses and untested exporters are recorded rather than silently filled. The negative coverage decision remains unchanged even if Zero-DCE++ is later admitted as an external-data zero-reference comparator.

baseline coverage insufficient
