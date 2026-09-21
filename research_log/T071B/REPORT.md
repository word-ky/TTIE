# T071-B — DONE / OFFICIAL_LOLV2_REAL_BASELINES_FROZEN

The matched official LOL-v2 Real comparison is complete. Each accepted baseline ran exactly once on all 100 T071-A official test lows, with no omissions, retraining, reference-conditioned inference or outcome-driven reruns. Final Ours was not rerun or modified. Both released supervised baselines outperform frozen Final Ours on matched PSNR and RGB-SSIM; this result does not authorize method tuning.

## Matched results

| Method | Mean PSNR (dB) | Median PSNR (dB) | Mean RGB-SSIM | Total inference seconds |
|---|---:|---:|---:|---:|
| Frozen Final Ours (T071-A, reused) | 18.53226686142791 | 18.19278833203082 | 0.5734772617839705 | 70.03712362400256 |
| Retinexformer | 22.79526194331298 | 22.91092489876291 | 0.8390188289794175 | 7.68263399918214 |
| SNR-Aware native_pad16 | 21.36590207395806 | 21.57562903944750 | 0.8495393435858116 | 2.53458334013703 |

Paired Ours-minus-baseline differences use per-image subtraction before aggregation:

- retinexformer: mean **-4.262995081885076 dB**; median **-5.081350353114713 dB**.
- snr_aware: mean **-2.833635212530162 dB**; median **-1.760276100712113 dB**.

## Provenance and training/exposure conditions

T033-A Retinexformer is reused unchanged: official source `caiyuanhao1998/Retinexformer` at `1e9a0efce4b306b6701b824768370ff26066c32a`, checkpoint `LOL_v2_real.pth` 6,478,393 bytes, SHA256 `539bd16c4da6179e45616329f249c4672951b1045193428e1d042c50d4b65a0b`. Exporter SHA `dc6649d2f577b3b6c0c458e4ff52d3956cb5eb3fd64385ec2b659737dbc4246c`; accepted config SHA `5260d0c65878f6a39712f70948be1936d8583531491d832cb59362fffba894ac`. Mode remains `default_no_gt_mean`, no ensemble, native RGB float32, factor4 reflect pad/unpad, clamp[0,1]. Ten accepted source/archive/checkpoint/config hashes checked before and after.

T045-A SNR-Aware is reused unchanged: official source at `1113144c82adc8bcc4a9ec27749ed75f196a4e4d`, checkpoint `LOLv2_real.pth` 156,523,164 bytes, SHA256 `432d29d370e9f674f1b6763d371b4c24569a86d21f0fd45a5797226274d85781`. Exporter SHA `72a8bdec919d1c35056f549aca3c2bfcca7a1a2960999a7bfa0727434a766b61`; config SHA `fcb29f50538cfd09ec425c83d7f2072477b24f7c2ab4f23507c3e1b37016b3fb`. Mode remains accepted `ttie_native_pad16`: native float32 RGB, low-derived 5x5 blur before padding, right/bottom reflect pad16, low-derived SNR, direct network, native unpad/clamp. This is the historical TTIE protocol adapter, not reproduction of resize-based official test4/paper numbers. Fifty-six accepted source/checkpoint/config hashes checked before and after. Parameter hash remains `11d3d667821719bd51e6e608c7876774b001643a28ed85193054bb45428190d4` before/after.

| Method | Training/exposure category | Published validation convention |
|---|---|---|
| Final Ours | Frozen development-fitted global selector/settings plus low-only per-image test-time optimization; not a supervised enhancement network and not wholly training-free | T070-A freeze before official test; no official-target adaptation/selection |
| Retinexformer | Official target-domain paired supervised LOL-v2 Real Train checkpoint | Pinned YAML explicitly uses official Test/Low and Test/Normal for validation |
| SNR-Aware | Official target-domain paired supervised LOL-v2 Real Train checkpoint | Pinned training YAML uses Train/Low and Train/Normal for validation |

These categories follow the accepted official checkpoint associations and hash-bound published recipes. Original training/selection logs are unavailable; we do not assert a verified history of actual checkpoint selection or that Retinexformer's official test was unseen upstream. No method was trained or fine-tuned in this cycle. The comparison matches evaluation inputs and metrics, not training regimes. Both baseline source trees remain external; no third-party source or checkpoint is redistributed.

## Frozen cohort, information boundary and evaluator

Exact T071-A pair manifest SHA `3fb94cbe7e81d37d719f106e5db5c144a48db062e51c3b2aa234555a1336fcfc`: all official IDs 690 through 789, native 400x600 RGB, same ordering and low/normal pairing. Canonical archive SHA `9820d8b112438d94d1f5d4d25817eee618a8cf4bc63a65cd6b19f5a98c3faefa` and all staged low hashes verified before execution. The runners consume only the unchanged accepted exporter CLI low list/checkpoint/config; cv2 decoding is restricted to those 100 lows, and PIL decoding is disabled during inference. No metric/quality table is parsed during exporter execution. No Ours outputs or clean/reference pixels enter baseline inference.

Both complete output tables precede first evaluation reference access at **2026-09-21T10:47:34.860494+00:00**:

- Retinexformer freeze: `2026-09-21T10:47:23.543068+00:00`; SHA `cffea2c485c4b7ca12a93448c5d85e705c45781d8a9de053e96239a95664e958`.
- SNR-Aware freeze: `2026-09-21T10:47:33.357272+00:00`; SHA `f9b97872d87c0367a8be95e04329a682b63cad207d7ea354b4dbc7ff737a49f9`.

Inference: 100 low decodes per method, reference reads 0, optimizer runs 0, model fits 0. Each manifest records input/output tensor and file hashes, native shape, runtime and memory. Postfreeze evaluator reads each official reference once and applies it to both saved outputs (100 total reference-member reads); separate verifier rereads those 100 references, without network inference. Ours metrics use immutable T071-A per-image SHA `eda432fbc802447ed273be37afba33118ceffa48a1c91925d07270446245be50`.

Primary metric arithmetic directly calls unchanged `research_log.T071A.core.metrics`. Reference uint8/255 is rounded to float32 before float64 metrics; output float32 is promoted to float64. PSNR is per-image -10log10(full RGB MSE), range1. RGB-SSIM: 11x11 Gaussian sigma1.5, K1=.01/K2=.03, population covariance, reflect half-sample symmetric borders, all RGB pixels/channels mean. No crop, resize, Y conversion, quantization or brightness matching. Separate verifier uses torch float64 MSE, explicit separable convolution SSIM, math.fsum/statistics mean/median and independently reconstructed paired deltas. All cohort/hash/ordering checks PASS; maximum numerical discrepancy **7.105427357601002e-15**, below 1e-11.

## Source, execution and validation

Source **`579c3691a80f5b7cadfd706a2fe6750876c53aa0`**, branch `codex/T071B-official-baselines`; authorization main `ab9e7df5c268f965bca2949b0f24ff51d321d8c0`. Task-owned changes only under `research_log/T071B/`: thin exporter wrapper, shared-metric evaluator, independent verifier, binding receipts, focused tests and evidence. Existing scientific/exporter sources are unchanged.

Run **`20260921-184705-ttie-t071b-official-baselines`** on NVIDIA RTX A6000 GPU0 (more free memory than GPU1 at launch), explicit release `/media/wenchang/F/wjq/TTIE/releases/20260921-ttie-t071b-official-baselines`, output `/media/wenchang/F/wjq/TTIE/runs/T071B-official-baselines`. Both sequential inference processes and evaluator/verifier exited 0; run finished `2026-09-21T18:48:10+08:00`.

Environment: Python3.12.12, torch2.4.0+cu121, CUDA12.1, cuDNN90100, numpy1.26.4, OpenCV4.11.0; accepted seed7 and deterministic cuDNN exporter settings. `CUBLAS_WORKSPACE_CONFIG=:4096:8`, OMP/MKL/OPENBLAS threads1. Exact commands/environment in `evidence/run/run.sh`. The workflow wrapper's `meta.json.releaseId` is historical/stale; explicit cd in its command and scientific `config.json.source_commit` identify the actual release. Original raw metadata retained, only line endings normalized in the repo copy.

Tests: baseline9 passed18.58s; final local12 passed17.27s; final remote12 passed1.78s (explicit pytest `--import-mode=importlib`); independent saved-output verifier PASS. Focused tests cover unchanged baseline padding/unpadding/clamp/SNR/input CLI, T071-A pairing/evaluator, actual official manifest provenance/order and paired median semantics.

Retinexformer peak allocation627518464bytes; SNR-Aware669319680bytes. Recorded synchronized inference time per image: Retinexformer .07682633999182144s; SNR-Aware .025345833401370328s. These accepted exporter timings exclude model load/disk decode/save, and include the first invocation. Retinex starts after input transfer; SNR includes blur/SNR/input transfer. T071-A Ours uses its frozen API timing. Do not interpret these as a rigorously identical end-to-end latency benchmark.

Observed pre-inference failures and repairs: minimal remote package initially caused pytest test_core name collision (fixed command import mode); preflight found archive SHA nested inside manifest.provenance (corrected key in preflight and verifier, added real-manifest test); one local test command ran from root rather than worktree (rerun from correct cwd). All resolved before any baseline official inference. No experimental failure, rerun, sample exclusion or scientific-setting change.

## Artifacts and next step

- raw: `/media/wenchang/F/wjq/TTIE/shared/t071b/T071B_raw.tar`, 576808960 bytes, SHA256 `48e1a32e779c744a65c83c1b7de9dd3a0951907b7ea141b5b62cfb9a21d7cea8`.
- recovery: `/home/wenchang/asdasdsad/wjq/TTIE/shared/t071b/T071B_recovery.tar.gz`, 127166 bytes, SHA256 `51d4334290f6a85af6c079111629e381c80cc0fc79dcf25754ee10a66279bb52`.

Full archive retains all 200 float outputs. Compact source/evidence/run archive is verified locally and on both server roots; committed evidence includes all 200 metric rows, frozen output manifests, input/output hashes, training/source receipts, reference-read timing, summaries and independent verification. Project-local state/report/HANDOFF preserve recovery context.

Return this immutable official comparison to the research lead. The measured Ours shortfalls are 4.263dB versus Retinexformer and 2.834dB versus SNR-Aware under their stated training regimes. No superiority claim is supported here. Stop this cycle; cross-dataset evaluation remains the next separately authorized task, and the priority lock prohibits Ours tuning now. This stacked evidence PR is for task-file review, not a recommendation to merge all historical branches.
