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
