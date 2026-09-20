# T062-C-R2 — DONE / NEGATIVE

On the single locked reference-unused100 cohort, the frozen step27 candidate retains a large mean gain but fails the preregistered worst-tail safety requirement. **Four of five gates pass; the qualification verdict is NEGATIVE.** No alternate step, cohort, selector, tuning or metric-driven rerun was attempted. This is internal fresh qualification, not a final benchmark claim.

Authorization `a557ccb68a17e03c30f579325bb916681d173993`; tested source `47bcab6bc9cbc23712ce503f758ccad6a57f240d`; branch `codex/T062CR2-fresh-step27`. Immutable T062-C-R1 cohort evidence52823001f99aa58548c0e490d05f52b33810b656, manifest SHA `e8a66a350bcce5282df2e8114355afc434bcd0f4ea69e54957ebb7618f9183c2`. Exactly100 paths/order retained.

| Gate | Measured | Requirement | Result |
|---|---:|---:|---|
| Mean PSNR delta vsT036 | 3.550431493935 dB | >=2.00 | PASS |
| Median PSNR delta vsT036 | 3.379578012346 dB | >0 | PASS |
| Regressions vsT026 | 10/100 | <=29 | PASS |
| Worst PSNR delta vsT026 | -7.334175346098 dB | >=-5.614 | FAIL |
| Mean RGB-SSIM delta vsT036 | 0.009839267210 | >=-.001 | PASS |

T062 vsT036 improve/regress/tie: **92/8/0**. Absolute paired means:

| Method | PSNR | RGB-SSIM |
|---|---:|---:|
| identity | 7.578243899154 | 0.163596040747 |
| T026 | 11.095192131607 | 0.405454515384 |
| T036 | 12.137533752746 | 0.402594534182 |
| T062 | 15.687965246682 | 0.412433801392 |

Worst T026-relative case: index66, Train/Low/low00221.png. T02622.832859726919dB; T03614.821924270353dB; T06215.498684380821dB. This postfreeze diagnostic does not select a new inference rule. Mean gains transfer to this cohort, but the frozen step27 rule does not satisfy its safety envelope.

Exact accepted T026/T036 trajectory modules and assets reused; both start independently from raw low/identity, Adam.03×40active and minimum T014-energy/earliest tie. T062 uses unchanged low-only gate, CommonRegion2/CommonBox12raw, accepted Lspa+10Lexp+5Lcol, target.60, pools4/16, Adam.03×27 and output27. Prefix test establishes bitwise equality of all28states/images/components and27gradients/pre-box states to the accepted T062-A trajectory. No inactive-all failure occurred. Canonical240 source/config bindings and all model hashes are in binding.json/config.json. No changes to frozen controls/loss/gate/asset/action settings.

All **300 method outputs** plus100 identity outputs and traces were frozen at **2026-09-20T01:33:30.465378+00:00**; first reference marker **2026-09-20T01:33:34.332667+00:00** (strictly later). Freeze SHA **700ae2612234eb139a20c3560b58c85dca1eef3849347c24ffc09575620c07de**. Inference extracts only100 low archive entries and records100low reads; reference evaluation is a separate process and reads only100 specified normal entries after all output hashes validate. Complete reference-open list and exact normal hashes are saved. Official-test/cross-dataset access0. This cohort is now reference-exposed and cannot be reused as fresh or recycled for tuning.

Independent separate verifier confirms immutable cohort,300 final output hashes, all trace hashes, initialization/27updates/step27, ordering,800 PSNR/SSIM values (including identity), aggregate means and all5 gates. Dot-product PSNR and separable-convolution SSIM maximum difference **7.74491581978509e-13**. Verification PASS confirms the scientific NEGATIVE result.

Validation:4 accepted-loss/prefix tests passed26.82s locally;7 affected control/prefix tests passed21.54s locally and3.30s on server. Existing dependency/deprecation warnings only. Sole run `20260920-092400-ttie-t062cr2-fresh` exits0; physical A6000GPU1; release `20260920-ttie-t062cr2-fresh`. Inference556.013750s, evaluation26.917855s. Full executed command and logs in evidence/run. Source committed/pushed before experiment. No experiment failures or reruns. The previously observed T062-A interpreter-cleanup guard issue was repaired by deactivating the guard after inference ends; adaptation reads remain restricted.

Artifacts: paired.csv/paired.json, config/cohort/code/model identities,100per-image receipts,300compact traces, globalfreeze, reference-open receipts, result and independent verification. Raw tensors remain in `/media/wenchang/F/wjq/TTIE/runs/T062CR2-fresh-step27` and fullraw archive `/media/wenchang/F/wjq/TTIE/shared/t062cr2/T062CR2_raw.tar`,1163724800bytes SHA `a7c7e9028fa94e4e3ae4db9167fc237d2a82ac1f21a58095176fc5110bbdd477`. Recovery4746177bytes SHA `dcaa1026d34f60da6c15ec842d586a40d15eec88122aab51aae5483131f801be` verified local/home/F; includes source, traces, receipts and runlogs. archives.json supplies paths/hashes.

Next: stop fixed-step27 qualification route and return to research lead. No official-test/cross-dataset evaluation, alternative k, second cohort, threshold relaxation or final baseline-gap claim. Lead-owned files unchanged.
