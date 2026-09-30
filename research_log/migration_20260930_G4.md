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
