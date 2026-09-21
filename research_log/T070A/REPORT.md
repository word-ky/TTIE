# T070-A: FINAL_OURS_CANDIDATE_FROZEN

The unchanged T067-B candidate is now packaged as degraded-image-only inference. All 200 full inference replays match accepted selected steps, states, output tensors and complete 27-update trajectory prefixes exactly. The predeclared first5development + first5transfer repeats also match exactly. Independent verification passed. This is a reproducibility freeze, not new qualification or a claim that the known exposed tail disappeared.

Source: `aa4d920dff4b5b76751c24266e95ac9696d55d90`; branch `codex/T070A-final-ours-freeze`; authorization `ce7337323d27eff1e2fc54ce8438a136d373514c`. Task-owned files are `research_log/T070A/{infer.py,manifest.py,constants.json,inference_binding.json,audit.py,verify.py,test_infer.py,binding.json,PLAN.md,authorization.md,scientific_state.md,REPORT.md,evidence/*}`. Review these files; the stacked historical branch is not a merge recommendation.

## Frozen inference

Scientific API: `FinalOurs(manifest).__call__(low)`. The only per-image input is the degraded native RGB float32 tensor. The manifest supplies only frozen global assets/configuration. There are no clean/reference images, labels, PSNR/SSIM, baseline results, oracle ranges, condition IDs or image IDs in the inference API. The wrapper recomputes gate and adaptation from the degraded image; stored traces and cohort anchors are used only by the separate audit.

It reuses the unchanged CommonRegion2/CommonBox12D EV/gamma/gain renderer, identity initialization, T062CR2 exact27-update T062 prefix, Adam lr.03 with accepted defaults, losses [L_spa,10L_exp,5L_col], original CLIP/prototypes/gate, T066A19D features and frozen model, threshold.5, rho.9857470621423519 and exact T067B lambda.875/earliest endpoint conventions. Feature arithmetic remains the accepted float64 path; adaptation stays float32. The energy/baseline model is not loaded.

The immutable `evidence/FINAL_OURS_MANIFEST.json` binds30 inference entries (module hashes plus constants/global resource descriptors), exact source SHA, four global asset hashes, fixed constants, native RGB preprocessing, seed7, TF32off, thread/CUBLAS settings and environment versions. All27 previously bound modules match their accepted T062/T063/T066/T067 hashes. CLIP/prototypes/gate match T036A_assets exactly; probability model SHA is `33f878dd090d5e85ba2d099c0e16bc7e4f1731b41f0c5fd90f2a8a5a30f62be0`. No resources were refit or substituted.

Manifest SHA256: `e7f129d931d27531e5f6a14cd72e8c94734c474c40c3c403959cc8f5764e5ab9`. It was generated before image replay; independently rebuilt content matches, and its hash stayed unchanged. Source commit refers to the executable wrapper; this later manifest/evidence commit avoids circular self-hashing. Runtime loading checks bound source/resources, constants and environment. Audit package binding has52 entries including target-free anchors; per-image anchors are not inference assets.

Operational entry point, using an already authorized degraded image under the frozen environment:
```
python -B -m research_log.T070A.infer --manifest /media/wenchang/F/wjq/TTIE/shared/t070a/FINAL_OURS_MANIFEST.json --low <degraded-image> --out <new-output-directory>
```
The entry point persists `output.pt` and the selected decision/state/output hashes. This example does not authorize opening a held-out image in this task.

## Replay and independent verification

| Check | Count | Mismatches |
| --- | ---: | ---: |
| Original development full inference | 100 | 0 |
| Already-exposed transfer full inference | 100 | 0 |
| Selected step/state/output | 200 | 0 |
| Complete trajectory prefix (states, gradients, proposals, components, values, gate/bounds, rendered outputs) | 200 | 0 |
| Predeclared repeat subset | 10 | 0 |
| Independent selected replay | 200 | 0 |

Development anchors are original T067B target-free candidate_freeze rows with lambda.875; transfer anchors are original T067C target-free choice_freeze. All28 rendered states,27 gradients/proposals and gate/bound fields are compared against the accepted traces. Replay table SHA256: `24fa03a778f0e082e8e94825ff4e2fec780ee2075cde7dc92729cf559569141b`.

The independent verifier renders all5,600 states from degraded images and bound accepted traces, recomputes target-free features and independent predictions/selection, compares original anchors and the primary trajectory receipts, checks all10 repeat receipts, and reproduces the manifest/result. It does not rerun the optimizer or trust the wrapper's selected checkpoint. Primary replay did rerun the complete image-only gate/optimizer on all200+10 images.

Accounting: primary optimizer_runs210, optimizer_updates5670 (27 each); independent optimizer_runs0; global model_fits0. `reference_reads=0` in primary and verifier. Reused ReadScope logs actual low-image/trace data reads. No clean/normal images, quality/safety labels, PSNR/SSIM, baseline outputs, new fresh data, official LOL-v2 Real test, LSRW or UHD-LL were opened. Frozen gate/model resources are global development assets, not per-image annotations.

## Tests, environment and receipt

Initial local relevant suite7passed31.43s; final local7passed21.79s. Remote7passed with2dependency deprecation warnings in2.52s. Warnings concern protobuf metaclass behavior under future Python3.14; no test failed. Tests cover forbidden API arguments, exact selection composition/constants and original27-update trajectory behavior. Command: `python -B -m pytest research_log/T070A research_log/T067B research_log/T062CR2 -q`. Independent verifier PASS after5,600GPUrenders.

Run `20260921-164629-ttie-t070a-final-ours`; start2026-09-21 16:46:35+08, primary complete16:49:28+08 (165.78s including model loading), verifier complete16:50:19+08; exit0. GPU1 NVIDIA RTX A6000; Python3.12.12, torch2.4.0/CUDA12.1, torchvision.19.0, numpy1.26.4, Pillow12.3.0, open_clip_torch2.26.1. Full exact environment in manifest; full commands in `evidence/run.sh`.

Release `/media/wenchang/F/wjq/TTIE/releases/20260921-ttie-t070a-final-ours`; output `/media/wenchang/F/wjq/TTIE/runs/T070A-final-ours`. Sequence: tests -> manifest freeze -> `research_log.T070A.audit` -> `research_log.T070A.verify`. Wrapper metadata contains a stale historical releaseId; explicit cd and source-bound config identify the actual release. Repository meta.json is LF-normalized; original bytes are retained in the recovery archive.

No scientific failures, mismatches, parameter changes or experiment reruns. A local receipt-summary parser initially assumed warning-free pytest output; its extraction was adjusted to record the actual summary. No scientific source/evidence changed as a result.

Raw archive 1095680 bytes, SHA256 `2b18839923b57edb4cb0cdcc758f0b233737fbaa11b5fd6952c8fdfab06d2f58`; recovery 2006029 bytes, SHA256 `3f8d3ece6b0eb96efeb41f89b5d34780f3d1e16080b47492db0595190f1e6237`. Recovery archive and manifest were verified locally and retained on both server filesystems. Exact paths in `evidence/archives.json`; model checkpoint assets remain at their bound existing server locations.

Next step: research lead reviews this frozen candidate and decides whether to authorize genuinely held-out evaluation in a separate cycle. Keep final datasets sealed now; do not adjust the known rare-tail behavior during this freeze.
