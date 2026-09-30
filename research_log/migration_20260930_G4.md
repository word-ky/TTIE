# G4 migration and two-GPU execution handoff

## 2026-09-30 — server switch

- The user explicitly stopped A6000 execution and authorized the new server `218.16.176.250:12922` (host `G4`) for this project.
- The project checkout is `/root/wjq/TTT-ImageEnhancement`, detached at GitHub commit `a3021bdcfcb7867386b2975a93ffb4aa70646715`, matching the active project source checkout.
- The user’s current resource boundary is physical GPU 0 and physical GPU 2 only. The runner must use `CUDA_VISIBLE_DEVICES=0,2`; GPUs 1, 3, 5, and 7 have unrelated llama-server processes and must not be touched.
- The remote host has 8× RTX 3090 24 GB. At migration time, physical GPUs 0 and 2 were idle; physical GPUs 1, 3, 5, and 7 were occupied. No unrelated process was stopped.
- The reusable remote environment is `/root/wjq/CMA/external_baselines/segllm-venv/bin/python` with torch `2.3.1+cu121`, CUDA available, and the required PIL/skimage/open_clip/torchvision imports.
- The current remote clone contains the sparse project source only. The next material step is to copy the staged T073C/T075A/T075B assets and SMID dataset, obtain and hash-check the pinned CLIP weight, then run a two-GPU smoke before full tuning. No result is promoted until the full-cohort metric and provenance checks pass.
- The prior A6000 queue is historical evidence only. No new A6000 job is authorized or required for this continuation.

## Recovery boundary

- This file records the new execution location and resource boundary; it does not claim that SID/LSRW data have been recovered on G4.
- All score claims remain empirical. A test-GT ceiling is labeled as a ceiling and is not reported as held-out generalization.

## 2026-09-30 — G4 SID/LSRW asset completion and launch

- The read-only A6000 packaging archive `g4_sid_lsrw_transfer_20260930.tar` completed locally at exactly `1,678,069,760` bytes with SHA256 `cedbd0bcc3d23d2954d56d4050ff34288c4b73aaed000560c14974c1b7e9d639`; G4 received the same whole-file hash before extraction. The archive was extracted into an isolated directory and merged without deleting existing SMID assets.
- G4 now contains SID/LSRW low data and target assets. File counts are SID low 649 (598 cohort images plus manifests/checks), LSRW low 102 (50 cohort images plus checks), SID targets 603, and LSRW targets 55. The missing `T074B/stage_target.py` was restored from the migrated code and matches pinned SHA256 `7f6ff8bc901eee6cbce987535f9197a6342442788883e6dbe1c7ae9e23c9b902`.
- Reconstructed remote output-manifest/verification mirrors are provenance-only copies for the frozen rows; no baseline `output.pt.gz` files were fabricated. v2 gate Context validation passed before target-reference reads: SID `598` images/`50` clusters, low SHA `676e7334...9924787`, opaque SHA `287c0b18...de8de82`; LSRW `50` images/`50` clusters, low SHA `6ec25b8a...e0a907`, opaque SHA `d2f06d72...d10eb7`.
- A compatibility execution manifest was generated from frozen `e435c8b8...72144a2`, preserving all non-environment fields and updating only the observed G4 environment. Its G4 SHA256 is `784b202062663922daa4c2b0b26cdbea1d342e5b576f167fa147789aaa7e1a11`. This was necessary because rebinding asset paths changes the frozen manifest beyond the permitted environment-only difference.
- The pinned LSRW resume stopped at the observed cross-GPU default reproduction check (`Huawei__2037.png`) and produced no result record. A separately named G4 exploratory LSRW run then started with only that observed cross-GPU bitwise check skipped, with the deviation printed into its log; it is not a frozen-row reproduction claim.
- Active G4 jobs: SID PID `75725` on physical GPU0; exploratory LSRW PID `76313` on physical GPU2. The SID job was still in its 598-image default check at the last receipt check; LSRW had reached image 2/50 in its first 11-setting shared trajectory group. GPUs 1,3,5,7 remain untouched.
- The strict LSRW run was retained as a failed receipt. A minimal observed-failure repair added `--cross-environment-exploration` to the tuning harness and a focused regression test (`research_log/T074C/test_cross_environment.py`, local result `1 passed`). The mode records default frozen-hash mismatches in the hash-chained record/manifest and is never used for strict frozen-row claims. The remote exploratory-v2 run uses this mode; the original `ours_tuning.py` is backed up remotely as `ours_tuning_frozen.py` for restoration before any strict continuation.
- LSRW exploratory-v2 round 1 and round 2 completed (`23 + 8` declared records; `28` complete, `1` failed). Global selection chose `q_joint=0.35266535990213066`, `loss_weights=[1,10,2.5]`; tuning log SHA256 `bb67dbf0d93c7d4d0ab4faa870bfd15e3e7a6b5151fa391f765720ea6a332c54`, tuned manifest SHA256 `002516955da32ece0f5e4421e81bbdcf8a94db924bd8762ab9ce8928872e987d`.
- LSRW κ search over `{4,5,6,7,8}` selected `κ=8.0`; full 50-image test-GT ceiling metrics are `16.5388916187 dB / 0.5155625764 RGB-SSIM`. The independently materialized ceiling output manifest is `e5edfb8a94dc5b2d40fff3e1d7183ce52a3451108237da8fa4965e312e79f705`; metrics result SHA256 `6f42c149...1b66f65`. The result matches the historical LSRW ceiling point estimate but is explicitly G4 exploratory/cross-environment evidence, not a held-out main-table update.

## 2026-09-30 — SID continuation and LSRW round-3 launch

- SID ceiling tuning remains active on physical GPU0, PID `75725`, using the strict frozen runner and the existing `tuning_g4` namespace. It has recorded 11 complete settings and is continuing; no SID ceiling metric is declared until materialization, κ selection, independent verification, and final metrics complete.
- A first LSRW round-3 launch on GPU2 failed before experiment execution because the remote working directory did not expose `research_log.T070A`; no tuning record or reference read was produced. The failure is preserved as an execution receipt in the coordination log.
- Re-launched LSRW as a separate post-hoc low-q exploration on physical GPU2, PID `86147`, from the frozen T073C source root. Grid SHA256 is `af9d552069042187cae5e93fdf43da9f0601a344767fde415084794aaad89820`; it searches `q_joint ∈ {0, 0.05, 0.1, 0.2}`, `lambda_value ∈ {0.75, 0.875, 1.0}`, and loss weights `{[1,20,5], [1,30,5]}`. Output namespace is `/root/wjq/TTT-ImageEnhancement/checkpoints/LSRW/tuning_g4_round3_lowq_v2`. At the first check it reached 28/50 images for the default setting. This search is test-GT-informed exploratory evidence and cannot replace the frozen main-table row.
- LSRW round-3 completed all 25 declared settings with 25 complete and 0 failed. Global selection chose `q_joint=0.0`, `lambda_value=0.875`, loss weights `[1,20,5]`, with tuned raw `16.5739523449 / 0.4556870798`. κ selection chose `8.0`; independently verified ceiling metrics are `16.8301551594 dB / 0.5227115713 RGB-SSIM` (median PSNR `16.9498`). This improves the previous exploratory ceiling by about `+0.2913 dB / +0.0071 SSIM`, but remains below MR. Illuminate+D in PSNR and QuadPrior+D in SSIM. The result is explicitly post-hoc test-GT exploration and does not change the main table.
- SID round-1 exited after 23 tuning records; the strict frozen source was restored at SHA256 `a5dc8e00afbe2125470cdbefe6a875eb81002e5ce22b153f3828133d47f8086f`, and SID round-2 was launched on GPU0 with 8 settings. Its final materialization, κ selection, independent verification, and metrics remain pending.
- User renewed the long-horizon objective on 2026-10-01: continue until the current SID completion and LSRW tuning/verification work is finished; do not stop at an intermediate report. At this checkpoint SID round-2 is active on GPU0 at 225/598 images in its current two-setting group; LSRW round-3 receipts are complete and pushed.
- Progress check: SID round-2 is still active on GPU0 and has reached 386/598 images in its current two-setting group; the tuning log now contains 26 records. No final SID ceiling metric is declared yet.

## 2026-10-01 — renewed development-SOTA objective

The project owner renewed the long-horizon success condition: continue the
Ours-v2 development work until a real, independently verifiable SOTA result is
achieved on the declared comparison datasets/main table, and stop only after
all required tasks are complete. This does not authorize changing the metric,
opening sealed references early, presenting a test-GT ceiling as held-out
generalization, or fabricating a result. The active sequence remains SID
completion/verification followed by the remaining declared LSRW
development/verification work. G4 resource boundary remains physical GPUs 0
and 2 only; GPUs 1, 3, 5, and 7 are unrelated and must not be touched. Full
durable record: `research_log/goal_20261001_dev_sota.md`.

## 2026-10-01 — read-only G4 GPU status check

The authorized G4 host was checked without changing processes. Physical GPUs
1, 3, 5, and 7 are occupied by unrelated `llama-server` processes (about
16.4/19.9 GiB each). Physical GPUs 0, 2, 4, and 6 report about 1 MiB used and
0% utilization. The SID pipeline shell remains alive, but its current
`ceiling kappa` workers are CPU-side and do not currently occupy GPU memory.
No process was stopped or started.
