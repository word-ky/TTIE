# T062-B — PASS: development-tuned global step 27

Exact classification: **the T062 zero-reference trajectory is useful and a global early stop rescues its safety/structure failure**. This is a development-selected global hyperparameter, not held-out efficacy or a final Ours-versus-baseline claim.

Safety-eligible steps are **17–27 inclusive**. The single prescribed rule (maximum mean PSNR delta among eligible steps, earliest ties) chooses **k*=27**. No other rule or per-image selection was evaluated.

| Measure at step27 | Observed | Required | Pass |
|---|---:|---:|---|
| Mean PSNR delta vs T036 | +3.472467140261 dB | >=+2.00 | Yes |
| Median PSNR delta vs T036 | +2.974555776439 dB | >0 | Yes |
| Regressions vs exact T026-A | 5/100 | <=29 | Yes |
| Worst PSNR delta vs T026-A | -5.580232304134 dB | >=-5.614 | Yes |
| Mean RGB-SSIM delta vs T036 | +0.011996745187 | >=-0.001 | Yes |

Absolute PSNR/SSIM **14.702507277274 / 0.358299061839**. PSNR improve/regress/tie vs T036 **94/6/0**. The worst-tail margin is only about0.034dB above its fixed limit; generalization remains untested. All41 rows, eligibility flags and every image/state metric are in `evidence/table.json` and `evidence/000.json` through `099.json`.

Authorization `b6f55c313b433fe75d3679ec7943f1f7774440c6`; tested source `9e735c8cffaee2fef1572b9c58bf5ba7a1ad45a9`. Input T062-A freeze SHA `301c041aedeb7d4925560a860417891e6bc64fa616da3c1c10565b10d3496d00`, cohort SHA `279c74b335999d6631d436c79ea80e9e2cccfc4b97cb7c0b992829d04cfaf40b`, baseline per-image SHA `5676541d245fdb41c54a543cf88a79ca37640eea94432af0665cca2ddce2277c`, prior selected metrics SHA `dfcf67ec1d427d1e2bdf775d3a0ad5eeb0b9c8d81bf735f0cd8bd8428cf54273`. Exact config/code/input hashes are in `binding.json` and `evidence/inputs.json`.

The existing100x41 images were read and scored directly. All frozen file hashes, individual rendered tensor hashes, cohort order and normal-image hashes matched; previous minimum-loss selected PSNR/SSIM reproduced within1e-10. No renderer, optimizer, CLIP, gate or objective was invoked. Development references were used solely offline under this explicitly authorized global tuning protocol.

Independent verification checks all4100 frozen state identities; every PSNR is recomputed by dot product and every SSIM by independent separable convolution. Maximum metric discrepancy **1.0871303857129533e-12**. A separate verifier reconstructs all41 aggregates and selection using80-digit Decimal means; maximum aggregate discrepancy **1.7763568394002505e-15** and exact k* agreement. No missing/nonfinite states or provenance failures.

Three focused tests passed18.20s locally and1.72s remotely. Sole run `20260920-054721-ttie-t062b-audit`, release `20260920-ttie-t062b-audit`, exit0. Eight CPU metric workers preserve existing SciPy definitions; runtime **71.154379s**. No GPU/model computation is needed for this saved-state audit. No failures, reruns or deviations. New adaptation/render/official-test/cross-dataset accesses all0.

Selection completed **2026-09-19T21:48:41.844118+00:00**, receipt SHA **`da7066dd446046215e6279a4dfce80c4982d50f69575c4e433e19bfec9297bf8`**. Table SHA `ed5ce25b0a44afab16c8ba074df1a10c3d7214261e876e63ae6d3775230e729a`. Recovery archive324,828bytes SHA `f179d7c8c354b5cf5cd2ba683534f3ef958a5596b106f94e81f084c7514249a1`, verified local/home/F; paths in `archives.json`. Reuses the immutable T062-A full raw archive rather than duplicating images. Commands/logs are in `evidence/run`.

Next: lead review of the frozen development candidate27. Do not silently promote it, rerun this unchanged task, choose another rule, or access held-out data before a new authorized task.
