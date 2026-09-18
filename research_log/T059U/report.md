# T059-U: fixed outer low-gradient abstention — negative

**low-norm abstention is too indiscriminate**.

Authorization `4df07dd6edc64a54efd98d81521b0c75635c877f`; source `f6f9a7b48743a7ee5ee500f26b91f4939e80d532`. Accepted E source830e80ba0e1a4d09bce9c9ffd57be162a781d013/head e15e91c4e9be401bcac6d3de039ec2d1ae7141fb6cec40c388d22549652d43d0; corrected C2 evidencef4be91c67a5bc0d5dec09ecbded4ad8877d192cc; S source371c1926e1dd8d6eae95ab04fe4a435e9db81388/evidence27d2c2d41c6a2aebfd1b16ab78b9bf2e0ad853e3.

## Result

Fixed tau literal **0.031453661388567547**. Gate acts iff float64 L2 norm of FP32 predicted gradient is >=tau; equality acts. Python JSON's shortest representation0.03145366138856755 denotes the identical binary64 value, not a threshold change.

Coverage **55/80=0.6875 <0.75** triggers the first scientific stop criterion. Ungated harms5; gate abstains on1/5, recall **0.2**; retains51/55 beneficial actions, **0.9272727272727272**;4 harmful acted anchors remain. These secondary observations also do not support safety transfer, but the official classification is the earlier coverage stop.

| Policy | Acted | Improved / harmed / tied | Mean A | Median A | p90 A | Max harm | Median A among acted |
|---|---:|---|---:|---:|---:|---:|---:|
| ungated | 80/80 | 55 / 5 / 20 | -3.7988555083610793e-05 | -2.2112635288531329e-05 | 0 | 7.305624691450463e-06 | -2.2112635288531329e-05 |
| gated | 55/80 | 51 / 4 / 25 | -3.7557350833975403e-05 | -1.9881414403812719e-05 | 0 | 7.305624691450463e-06 | -3.6258893235260881e-05 |

A is full-RGB absolute MSE_after-MSE_state0. Mean/median/p90 include all80 banks; acted median uses acted subset. No baseline-relative metric or threshold adjustment.

## All ungated harms

| Bank / image / row | Norm | A ungated | Gate abstained |
|---|---:|---:|---|
| 28 / 37670 / 419 | 0.034492605929588603 | 1.3811400336183333e-06 | False |
| 29 / 37670 / 443 | 0.062088655414304104 | 2.7341289968452198e-06 | False |
| 54 / 38070 / 905 | 0.096214263805117717 | 7.305624691450463e-06 | False |
| 126 / 39951 / 2127 | 0.019952157446661022 | 5.239592599931564e-06 | True |
| 300 / 45728 / 5452 | 0.038378106536778901 | 4.2847073214158567e-07 | False |

## Cohort and information order

Exact published C2 split uncompressed SHA **f0fa4c44d96fb91a7d6d0047b9d06750becc01cba25019875bd277640bb11242**, recovered from pinned published gzip Git blob2fa7be41c1824f54681bb2a3667e27f06ee733cc. All16 images/80 banks/1460 global rows exactly equal E's excluded outer partition; zero row/image overlap with E inner development. Only80 unique state0 anchors receive actions. This is separate source-cohort validation, not target-domain deployment or a claim of never-before-used C2 evidence.

Image IDs: [36660, 37670, 38070, 38825, 39551, 39951, 41488, 41990, 42563, 43435, 44195, 45070, 45728, 46463, 47121, 47801]

Bank IDs: [0, 1, 2, 3, 4, 25, 26, 27, 28, 29, 50, 51, 52, 53, 54, 75, 76, 77, 78, 79, 100, 101, 102, 103, 104, 125, 126, 127, 128, 129, 150, 151, 152, 153, 154, 175, 176, 177, 178, 179, 200, 201, 202, 203, 204, 225, 226, 227, 228, 229, 250, 251, 252, 253, 254, 275, 276, 277, 278, 279, 300, 301, 302, 303, 304, 325, 326, 327, 328, 329, 350, 351, 352, 353, 354, 375, 376, 377, 378, 379]

Same S target-free CPU head-input-gradient convention on the fixed1460-row outer batch, accepted cached degraded-image Jacobians, train-only E normalization and literal inherited T054 renderer/Adam lr.05/betas.9,.999/eps1e-8. All80 raw/y0/detail/mask/grid states match accepted T059A. Physical A6000 GPU1 executes80 ungated comparison steps. Gated policy reuses identical v/y1 for55 passing banks and zero/y0 for25 abstentions; **no second optimizer call**. All80 q/g/norm/decision/ungated+gated displacement/output tensors fsynced and hashed.

Freeze **2026-09-18T17:34:15.816120+00:00** precedes first outer source-reference access **17:34:25.401387+00:00**. Separate evaluation and independent verification use only16 unique pinned outer source clean images. No reference gradients or stored MSEs used; direct frozen-image RGB MSE recomputed. Reference reads never alter gate/action.

Counters: training_runs0,new_heads0,head_optimizer_steps0,ungated_comparison_steps80,gated_acted55,premature_outer_reference_reads0,outer_source_clean_images16,reference_gradient_reads0,target_domain_access0,lolv2_access0,official_test_access0,inference_reference_leakage0. Outer source reference evaluation is explicitly authorized for U after freeze.

## Validation and artifacts

**5 tests passed4.93s**: exact threshold/equality and ordered gates plus inherited T054 first-Adam/blur/zero-gradient/inactive identity tests. Sole run `20260919-013345-ttie-t059u-outer`,2026-09-18 **17:33:49–17:34:43UTC**,exit0. Independent NumPy/SciPy all80 gate/norm/Adam/renderer/MSE/policy-statistic/classification replay PASS. Maximum Adam error **1.4612744791975274e-8** (<1e-7); render error **3.7737047930974654e-8** (<1e-6). Gated output is byte-identical to ungated output or y0 as prescribed. All199 source bindings, input files and selected Jacobian storage ranges unchanged. No scientific failures, code repair, rerun or deviation. Existing NVML warning nonblocking; local D full, so remote project and memory-only API delivery are authoritative.

Commands: pytest for U/S focused tests; separate act.py -> evaluate.py -> verify.py processes with fixed CPU threading and physicalCUDA1. Complete80 table, split, actionvectors, fullfield, head normalization, freeze/input/source hashes and independent receipts are published. Large80 image/output tensors remain in dual-disk home/F archives listed in archives.json with member manifest; compact source/evidence recovery listed in recovery.json. Threshold stayed immutable.

Stop and await research-lead review. No threshold search, optimizer change, second step, retraining, target-domain/LOL-v2/official-test access, rollout or self-merge.
