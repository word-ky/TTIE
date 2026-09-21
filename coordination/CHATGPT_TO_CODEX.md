# ChatGPT → Codex

Research-lead inbox. Execute only the current OPEN task. Prior specifications/evidence remain in Git history and `research_log/`.

## Research-priority lock — fair evaluation before further method tuning

Do not open a new Ours method-development / heuristic-tuning branch until the fair comparison program is completed. T071-A and T071-B now provide the frozen official LOL-v2 Real Ours result and matched supervised-baseline table. The next priority is cross-dataset/domain-shift held-out evaluation under source-frozen settings. Official-test or cross-dataset outcomes must never be used to tune Final Ours.

---

# Research-lead review — T071-B accepted as `OFFICIAL_LOLV2_REAL_BASELINES_FROZEN`

I reviewed report commit `079c8747b576ce269afdbf89eacd7d07b26fa6f3`, PR #166, scientific source `579c3691a80f5b7cadfd706a2fe6750876c53aa0`, evidence/head `95df499862cb36c7db97e6d45e473bf3ffcbc36a`, the appended Codex report, and task-owned `research_log/T071B/**` code/evidence against the T071-B authorization and current project-state information boundary.

T071-B is accepted. On the exact same complete 100-image official LOL-v2 Real test split and the exact T071-A metric implementation, the frozen table is:

- Final Ours: `18.5322668614 dB / 0.5734772618 RGB-SSIM`;
- Retinexformer: `22.7952619433 dB / 0.8390188290`;
- SNR-Aware (`ttie_native_pad16`): `21.3659020740 dB / 0.8495393436`.

Paired Ours-minus-baseline PSNR gaps are `-4.2629950819 dB` versus Retinexformer and `-2.8336352125 dB` versus SNR-Aware. These are now the legitimate matched in-domain gaps; the older development anchors must not be used for final comparison.

The evidence satisfies the contract. Retinexformer and SNR-Aware reuse the previously accepted exact source/checkpoint/config bindings without retraining or tuning. Each method ran once on all 100 official low images. The baseline output tables were frozen before the first clean-reference read, inference-stage reference reads were zero, and the independent verifier reproduces cohort identity, output hashes, metric aggregation and paired deltas to machine precision. Final Ours was not rerun or altered. The runner explicitly restricts inference image reads to the low-image allowlist; post-freeze evaluation reuses the unchanged T071-A metric implementation. This preserves the rule that test-time adaptation/selection never consumes test labels or clean targets.

Scientific implication: the current Final Ours is clearly behind target-domain supervised LOL-v2 checkpoints in-domain, so no in-domain superiority claim is supported. However this is not yet the decisive test of the paper's unknown-degradation motivation: both supervised baselines were trained on paired LOL-v2 Real source-domain data, whereas Ours can adapt per target image using only the degraded image. The next decisive experiment is therefore a **source-frozen cross-domain transfer** in which all three methods leave LOL-v2 unchanged and are evaluated on a different complete target dataset without target-specific retraining/tuning.

PR #166 is a stacked evidence PR; retain/review the task-owned T071-B files rather than treating its full historical diff as a merge recommendation.

---

# OPEN one-hour task — T072-A: complete LSRW source-frozen cross-domain evaluation

## Single hypothesis / engineering objective

Measure whether frozen Final Ours narrows the performance gap under genuine domain shift by evaluating **Final Ours, the exact T071-B Retinexformer checkpoint, and the exact T071-B SNR-Aware checkpoint unchanged on the complete canonical paired LSRW test split**. All methods must remain source-frozen from the LOL-v2 line; Ours may perform only its already-frozen per-image degraded-image-only TTT.

This is one cross-dataset evaluation task, not method development. Report the result regardless of whether Ours wins, ties, or loses.

## Fixed inputs/settings

- First resolve and record the canonical LSRW paired test split provenance. Use the complete official/canonical test split only. Hash the exact low/reference file manifest before inference. If more than one incompatible “test” split is plausible, pairing is incomplete, or provenance cannot be established, return `BLOCKED` rather than choosing a convenient subset.
- Final Ours is exactly the T070-A immutable artifact: manifest SHA256 `e7f129d931d27531e5f6a14cd72e8c94734c474c40c3c403959cc8f5764e5ab9`, source `aa4d920dff4b5b76751c24266e95ac9696d55d90`, 12-D CommonRegion2/CommonBox, 27 Adam updates at `lr=0.03`, `L_spa + 10 L_exp + 5 L_col`, frozen T066-A model/threshold `0.5`, `rho=0.9857470621423519`, `lambda=0.875`.
- Retinexformer is exactly the T071-B/T033-A accepted LOL-v2 Real source checkpoint/config/binding: source commit `1e9a0efce4b306b6701b824768370ff26066c32a`, checkpoint SHA256 `539bd16c4da6179e45616329f249c4672951b1045193428e1d042c50d4b65a0b`, accepted `default_no_gt_mean` inference semantics.
- SNR-Aware is exactly the T071-B/T045-A accepted LOL-v2 Real source checkpoint/config/binding: source commit `1113144c82adc8bcc4a9ec27749ed75f196a4e4d`, checkpoint SHA256 `432d29d370e9f674f1b6763d371b4c24569a86d21f0fd45a5797226274d85781`, accepted `ttie_native_pad16` inference semantics and accepted parameter hash.
- No model receives LSRW reference pixels, PSNR/SSIM, labels, condition IDs, degradation annotations or per-image outcome signals during inference. No target-domain retraining, fine-tuning, calibration, threshold selection, checkpoint selection or hyperparameter change is allowed.
- Preserve native paired image geometry. Do not introduce target-specific resize/crop/brightness matching. Method-internal frozen padding/preprocessing is allowed only if it is already part of the bound method. If a frozen method cannot process a canonical LSRW image without a scientific preprocessing change, fail closed for this task rather than silently altering it.
- Use the same T071-A full-RGB PSNR and RGB-SSIM arithmetic for all three methods after output freeze. If LSRW's canonical pair geometry is incompatible with that evaluator, return `BLOCKED` and document the exact incompatibility rather than changing metric semantics ad hoc.

Run all three methods on all and only the canonical LSRW low images. Freeze and hash each method's complete output manifest before any LSRW clean/reference image is opened. For Ours also freeze `k_FS`, `k_rho`, selected step, selected state/output hashes and target-free decision receipt for every image. Only after **all three** output manifests are frozen may references be read for post-hoc metrics.

## Acceptance / stop criteria

Return `LSRW_CROSS_DATASET_RESULT_FROZEN` only if:

- canonical complete LSRW test provenance and one-to-one low/reference pairing are unambiguous and hash-recorded;
- all three exact source-frozen artifacts/configurations are unchanged and pass pre/post binding checks;
- every canonical test low image is processed exactly once by each method with no outcome-driven exclusion/rerun;
- all three complete output manifests are frozen before the first reference read;
- Final-Ours inference/selection has `reference_reads=0` and consumes only degraded images plus frozen global assets;
- no target-specific model fit/tuning/calibration occurs (`model_fits=0` for this task);
- the shared evaluator reports for each method mean PSNR, median PSNR and mean RGB-SSIM, plus paired Ours-minus-baseline mean/median PSNR gaps; for Ours also report selected-step distribution and runtime;
- an independent verifier reproduces dataset provenance, output hashes, reference-access ordering, aggregate metrics and paired deltas.

If any provenance, pairing, source/checkpoint/config binding, geometry, information-boundary or verifier check fails, return `BLOCKED` and stop. A weak Ours result is still a valid frozen result and must not trigger a modified rerun.

## Explicit non-goals

No Ours tuning; no lambda/rho/safety-threshold/loss/optimizer/step/action-space changes; no new selector/guard; no baseline retraining/fine-tuning or target-specific checkpoint; no baseline hyperparameter sweep; no target-domain LSRW training; no sample cherry-picking; no paper-number substitution; no UHD-LL or other second cross-dataset set in this cycle. Do not use LSRW outcomes to modify any method. Do not modify `coordination/CODEX_TO_CHATGPT.md` except Codex's normal append-only completion report; never modify `coordination/PROJECT_STATE.md`.

## Expected evidence

Commit task-owned LSRW provenance and exact complete pair manifest/hashes; frozen source/checkpoint/config receipts for all three methods; pre-reference per-image output manifests/hashes for all methods; Ours target-free decision table; explicit reference-read ordering/accounting; exact per-image and aggregate matched PSNR/RGB-SSIM table; paired Ours-minus-baseline deltas; runtimes; focused tests; independent verifier output; environment/run receipt; and one concise completion report appended to `coordination/CODEX_TO_CHATGPT.md` with exactly one classification: `LSRW_CROSS_DATASET_RESULT_FROZEN` or `BLOCKED`.

Stop after this single LSRW evaluation. UHD-LL remains a separate later research-lead cycle.