# T059-V: online target-free Jacobian/action bridge — pass, narrow coverage

**online target-free Jacobian bridge is reproducible on source anchors**. All16 prescribed anchors pass all fixed bounds. This is a reproducibility result only, not a performance or safety result.

Authorization `9844a84eb58e5e6ca64d4366ccf00ca43fdafe50`; source `b4bd9ee9e2df829b915a0494c50ed1642bcfa102`; comparison U evidence `88de78b1a91adfc6dec71c378cdf8a4ec044cb51`. Exact E checkpoint e15e91c4e9be401bcac6d3de039ec2d1ae7141fb6cec40c388d22549652d43d0 and train-only normalization unchanged. Original T054/S renderer/Adam, T059A feature Jacobian VJPs and fixed CLIP/prototypes/gate source definitions reused and hash-bound.

## Result and scope limitation

**15/16 selected anchors have a closed gate, zero J, zero g and identity action. Only bank300/image45728 has a nonzero Jacobian/action.** The prescribed lowest-bank-per-image selector gives clean-condition source-side banks. These authorized bank state0 tensors were used as inputs; no separate clean/reference JPG was opened. Do not generalize this result to broad nonzero-gradient coverage or degraded target-domain performance. No cohort substitution or additional anchors were run.

Maximum feature absolute error **4.887580871582031e-6 <=5e-5**. Maximum J relative Frobenius error **1.0964620047637406e-6 <=2e-3**. Minimum gradient cosine **0.9999999999994275 >=0.999**; maximum gradient relativeL2 **1.0740708862065158e-6 <=1e-2**. Maximum online/cached output error **5.960464477539063e-8 <=1e-4**. Maximum v1 absolute error7.450580596923828e-9; q absolute error3.5762786865234375e-7. All16 y0 and masks exactly match cached U state0 inputs.

For exactzero pairs, declared equality convention is relativeerror0/cosine1; they are separately reported as zero cases. One-zero cosine0; relative denominator floor1e-30 only defines zero-reference case and never loosens a nonzero comparison. The sole nonzero bank300 passes every numeric criterion without these conventions. No tolerance changes or differentiation fallback.

| Image / bank / global row | Active regions | Feature maxabs | J relative | g cosine | g relative | Output maxabs |
|---|---:|---:|---:|---:|---:|---:|
| 36660 / 0 / 0 | 0 | 3.5762786865234375e-06 | 0 | 1 | 0 | 0 |
| 37670 / 25 / 370 | 0 | 2.5033950805664062e-06 | 0 | 1 | 0 | 0 |
| 38070 / 50 / 855 | 0 | 1.430511474609375e-06 | 0 | 1 | 0 | 0 |
| 38825 / 75 / 1317 | 0 | 2.2649765014648438e-06 | 0 | 1 | 0 | 0 |
| 39551 / 100 / 1664 | 0 | 2.86102294921875e-06 | 0 | 1 | 0 | 0 |
| 39951 / 125 / 2126 | 0 | 4.0531158447265625e-06 | 0 | 1 | 0 | 0 |
| 41488 / 150 / 2634 | 0 | 4.8875808715820312e-06 | 0 | 1 | 0 | 0 |
| 41990 / 175 / 3119 | 0 | 3.5762786865234375e-06 | 0 | 1 | 0 | 0 |
| 42563 / 200 / 3604 | 0 | 4.5299530029296875e-06 | 0 | 1 | 0 | 0 |
| 43435 / 225 / 4066 | 0 | 2.1457672119140625e-06 | 0 | 1 | 0 | 0 |
| 44195 / 250 / 4528 | 0 | 1.4901161193847656e-06 | 0 | 1 | 0 | 0 |
| 45070 / 275 / 4967 | 0 | 3.7848949432373047e-06 | 0 | 1 | 0 | 0 |
| 45728 / 300 / 5452 | 2 | 1.430511474609375e-06 | 1.0964620047637406e-06 | 0.99999999999942746 | 1.0740708862065158e-06 | 5.9604644775390625e-08 |
| 46463 / 325 / 5983 | 0 | 2.2649765014648438e-06 | 0 | 1 | 0 | 0 |
| 47121 / 350 / 6422 | 0 | 2.0265579223632812e-06 | 0 | 1 | 0 | 0 |
| 47801 / 375 / 6884 | 0 | 2.5033950805664062e-06 | 0 | 1 | 0 | 0 |

## Online independence and ordering

Selection.json binds exactly the fixed U16 images and lowest bank per image before execution. Online process opens only selected source-side bank_images.pt, frozen model/prototypes/configuration and E head. It does **not** open bank.pt, bank_decisions.json, cachedx/J/q/g/v/y, Uevaluation_table/result, reference gradients or clean-image files. PIL.Image.open is disabled in that process. FixedObjective recomputes original gate scores/evidence from each image; CommonRegion2 starts at zero raw parameters; inherited Detail reconstructs y0/grid/mask. FrozenCLIP+prototypes compute fresh differentiable28-D features, original A autograd computes8 nonconstant feature-coordinate VJPs (remaining20 constant coordinates zero). ECPU input-gradient q is evaluated peranchor; fresh online J^Tq forms the gradient; exact inherited Adam lr.05/betas.9,.999/eps1e-8 forms one action. Cached values never enter this path.

All16 x/J/q/g/v1/c1/y0/y1 tensors persisted/fsynced by **2026-09-18T18:22:47.646784+00:00**. First cached comparison read **18:22:52.258505+00:00**, separate process. Published U Git blob identities and frozen action SHA256s are verified only for comparison. Independent verifier runs afterward, with no model/feature re-execution. Source209 bindings, online inputs, frozen online outputs and cached comparison inputs remain unchanged.

Frozen CLIP checkpoint1bd3c7172de5b207ceac554f5ab5266166f3b9baccc9af5989bc801016d080ad; prototype b4b32dbd96c65dcf606ee38d7450ebf348f5731823503b9c71ba15ec78217ac7; T014 model/gate config1db3e26bb3fcc68bc3212f7189cfb71cef279612b37ff2de0c382937efbc13b3. The configuration contains fixed source-trained gate calibration, not an evaluation target. Full file bindings, normalization and per-tensor hashes in selection/source_binding/online_freeze/online_records.

Counters clean_reference_reads0,reference_gradient_reads0,target_domain_access0,lolv2_access0,official_test_access0,inference_reference_leakage0,cached_action_tensor_reads_before_freeze0. Online16 feature/J evaluations plus16 gate image forwards;16 inherited detail optimizer steps, zero head training. Physical A6000GPU1 for CLIP/autograd/render; CPU frozen head matches S/U device convention. No clean/MSE/PSNR/SSIM evaluation.

## Validation and delivery

**6 tests passed4.35s**, covering zero equality, exact bounds, signed-gradient errors and inherited T054 first-Adam/blur/zero-gradient/inactive identity. Sole run `20260919-022225-ttie-t059v-online`,2026-09-18 **18:22:29–18:23:05UTC**,exit0. Independent scalar-loop tensor error/cosine computation plus NumPy chain/Adam and SciPy renderer reconstruction verifies all16 comparisons and classification. No scientific failure, repair, tolerance patch, alternate differentiation method or rerun. Existing NVML warning nonblocking; local D remains full, so remote project and GitHub preserve outputs.

Commands: pytest for V/S; separate online.py -> compare.py -> verify.py with physicalCUDA1, TF32off, fixedseed7 and single CPU threads. Full16-row table, online field tensors, per-anchor output hashes, timestamps/input/cache bindings and independent receipts committed. Large output tensors in home/F evidence archive with member manifest; compact source recovery in recovery.json. No prior report or lead-owned file modified.

Stop for review. Any target-domain development rollout requires the next explicit research-lead task. This pass alone authorizes no rollout and establishes no enhancement-quality or safety benefit.
