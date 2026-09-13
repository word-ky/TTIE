# T022-A DONE — ready for review

# T022-A untuned LOL-v2 Real validation anchor

UTC: 2026-09-13T18:49:24.077647+00:00

Verdict: **benchmark-ready**. Exact accepted T014/Ours-Core ran on all100 predeclared validation images at native600x400 resolution, on A6000 GPU1. Quality remains weak in absolute terms; this is a reproducible untuned anchor, not a SOTA or official-test result. No performance threshold or tuning was applied.

| Metric | Raw mean | Ours mean | Raw median | Ours median |
|---|---:|---:|---:|---:|
| PSNR | 8.109722671658 | 9.272868945613 | 7.600161540858 | 8.680092886101 |
| SSIM | 0.160022843480 | 0.273006012336 | 0.138977265500 | 0.242630539370 |

All100 images were active, none abstained, all executed40 updates (4000 total). Mean selected checkpoint step 26.95. GPU wall time per image: mean 2.300624167872s, median 2.430337020516s, p95 2.477017533765s. Timing encloses the unchanged T014 trajectory with CUDA synchronization; excludes model loading, input decoding/transfer, output writing and metric computation. First-image inference is retained, no benchmark-image warm-up or retry. GPU is NVIDIA RTX A6000; runtime versions are in the run config.

## Dataset and predeclared split

Canonical author source: [flyywh/CVPR-2020-Semi-Low-Light](https://github.com/flyywh/CVPR-2020-Semi-Low-Light), whose LOL-v2 link points to [LOL-v2.zip](https://drive.google.com/file/d/1dzuLCk9_gE2bFF222n3-7GVUlSVHpMYC/view?usp=sharing). Original archive1046491030 bytes, SHA256 `9820d8b112438d94d1f5d4d25817eee618a8cf4bc63a65cd6b19f5a98c3faefa`. No mirror, relabeling, resizing or split mixing.

Archive root `LOL-v2/Real_captured/` contains689 Train pairs and100 Test pairs. Pair mapping `Train/Low/lowNNNNN.png` to `Train/Normal/normalNNNNN.png` (analogous Test paths); all789 pairs have matching600x400 PNG-header dimensions. Synthetic archive members were not used. Counts, byte hashes and dimensions for every real pair are in `T022A_data/dataset_manifest.json`.

Validation is first100 ascending SHA256(`TTIE-T022A-seed7|` + relative low path), where paths are explicitly relative to `LOL-v2/Real_captured/`, e.g. `Train/Low/low00653.png`. `filename_binding.json` was written before reading image payloads; subsequent inspection of normal files only hashed encoded bytes and read PNG IHDR, without decoding pixels. Final split SHA256 `b88c8347005984b5523b117b52c0c068672fe172eb7c9aa5b60102d350e2d85b`. Remaining589 training pairs were not used for inference/training/scoring. Official100 test pairs were manifest-only, never deployed to the inference input or scored.

## Exact method reuse and isolation

Scientific source `10aafb817af1f6405f0a086e5852fadc56c25f6d`; post-freeze evaluator `25b7b285cbc0d94c5f2f6d9e25f1b61922446658`. Every preexisting ttie file from accepted T014 source `f861b2c6ffde6d017cb174ef8e00cb75701bf5e1` is unchanged. The native loader is the only new input operation; it performs RGB float32 division by255 without resize. `ttie.energy_ttt.trajectory` is called directly with image/scorer/receipt/head, basis region2 and max_steps40, exactly as the accepted Ours branch; unrelated comparison methods are not executed. Accepted CLIP, learned prototypes, T007 abstention gate and Sobolev energy checkpoint SHA identities are verified in `T022A_assets.json` and persisted run config. Adam lr.03, hard Region2 EV+gamma, action projection, identity/reset,40 active updates and minimum predicted-energy selection are unchanged. No T019 selector, retraining, parameter search or reference-informed retry.

Only100 low-light images were initially deployed. Every output tensor (image/raw/grid), selected step, gate, energies/projections/diagnostics and state/score/feature trajectory was persisted and hashed. All-output freeze completed `2026-09-13T18:44:59.886126+00:00`; only afterwards, at `2026-09-13T18:45:55.631932+00:00`, were selected normal images deployed. The inference process records exactly100 low-only Image.open calls and has no normal-root argument. Static call audit confirms only label-free original trajectory arguments. Independent post-freeze audit checks all hashes, split, sizes/finite values,40-or0 update counts and minimum-energy checkpoint decisions before reference decode. Reference deployment receipt and unchanged freeze hash are preserved.

## Metrics and verification

Per-image RGB PSNR=-10log10(MSE), float64 arithmetic over float32 [0,1] decoded pixels, native full frame. SSIM is the unchanged accepted T021 convention: RGB, Gaussian11x11 sigma1.5, population covariance, K1.01/K2.03, reflect border, full-map mean, no crop/Y conversion/per-image normalization. Optional LPIPS was not installed and is omitted without runtime changes.

Independent PyTorch PSNR and explicit-separable-kernel SSIM agree with primary metrics across all100 raw/Ours pairs; maximum error 3.5527136788005009e-15. Local independent aggregation reproduces all means/medians exactly. All metrics and saved image tensors are finite. Executed inference/evaluation source hashes match local committed files; all historical T014 code remains byte-identical in Git.

Tests:9 baseline tests passed16.16s;2 native-loader/split tests passed6.96s; final affected suite13 passed14.72s, plus independent-kernel check and actual100-image end-to-end audit. Commands: `python -m pytest tests/test_lolv2_core.py tests/test_ssim_transfer.py tests/test_energy_core.py tests/test_sobolev_core.py -q`; `python -m ttie.lolv2_core --low-root <low-only> --split <split.json> --assets <assets.json> --out <audit>`; after freeze, `python scripts/evaluate_t022a.py --audit <audit> --low-root <low> --normal-root <normal> --split <split> --deployment <deployment.json>` (PYTHONPATH points to project root).

## Artifacts and observed failures

Run `20260914-024025-ttie-t022a-core`. Compact evidence under `research_log/remote_runs/20260914-024025-ttie-t022a-core/`; includes per-image CSV, timing, decisions, trajectories, aggregate JSON, freeze/config, logs and independent audit. Original output.pt files remain on the server under the same run path; compact Git package deliberately omits large image tensors but retains their hashes and exact index paths. Original canonical archive is project-local `.autodl/LOL-v2.zip`. Complete run/input/source backup: `/media/wenchang/F/wjq/TTIE/shared/t022a/T022A_execution.tar`,11973048320 bytes, SHA256 `55baa3bce132b8dfa2b82dab7a12872e5eb6308398faa7540a30cf786ec723b9`; original100 output hashes verify within that archive.

A transient SSH timeout/log disconnection and one GitHub push timeout recovered without restarting inference. One PR creation attempt returned422 because that push had failed; successful push then created PR42. Missing root-path gate JSON was handled by deploying the exact accepted Git blob before running. No scientific execution or metric failure; no outputs/settings changed after results. No unrelated jobs changed.

Output serialization retains underlying PyTorch trajectory storage (11808423700 bytes across100 output.pt files). Selected tensors and their metrics are correct and independently verified; frozen files are preserved unchanged. Slow gzip backup was stopped and its partial archive retained; the uncompressed F-drive backup is complete and all100 original output hashes were independently checked inside it. This is a storage overhead issue, not an inference or metric failure.

Recommended next step: research lead reviews this weak but valid untuned real-benchmark anchor and scopes the next validation-only tuning or baseline-comparison task. Stop here; no corrective tuning or official-test evaluation in T022-A.

benchmark-ready
