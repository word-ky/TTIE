# T059-D — C2 value failure is not explained by bankwise additive offsets

All five frozen C2 aggregate statistics replay exactly (zero deltas), then the fixed state0 bank-anchor diagnostic reduces aggregate value Huber from **0.19325308501720428** to **0.10301457345485687**. The latter still exceeds **0.07650849781930447**, with gate margin **-0.026506075635552406**. This diagnostic does not reverse the C2 negative: global detail median cosine remains0.4797055721282959 below0.50. No training or model change occurred.

## Bank-relative evidence

Exactly80held-out banks/1,460rows and one unique state_index0 anchor per bank. Allrows, including80anchors, contribute to the row-weighted aggregate. The80-bank unweighted mean relative Huber is0.07833400087765767; it is descriptive and is not substituted for the specified aggregate. The original float32 standardized log-MSE convention is preserved, including log in double then float before head normalization.

| Per-bank diagnostic | Defined / undefined | Mean | Median | P10 | P90 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| unanchored_huber | 80 / 0 | 0.661383376 | 0.0813585259 | 0.00871815467 | 2.04007528 | 0.00175139366 | 5.96369267 |
| relative_huber | 80 / 0 | 0.0783340009 | 0.00896328688 | 0 | 0.0639944568 | 0 | 4.56514597 |
| spearman | 60 / 20 | 0.840535019 | 0.961287516 | 0.770782609 | 0.986168217 | -0.645217391 | 1 |
| argmin_regret | 80 / 0 | 0.105623739 | 0 | 0 | 0.0420367599 | 0 | 6.04692793 |
| anchor_residual | 80 / 0 | 0.3431726 | 0.0719535649 | -0.961475933 | 1.78961205 | -4.59181356 | 7.40751314 |
| legacy_positive_fraction | 60 / 20 | 0.950694443 | 1 | 0.875 | 1 | 0.458333343 | 1 |
| legacy_median_cosine | 60 / 20 | 0.858576602 | 0.939500272 | 0.667386794 | 0.979935288 | -0.0775152743 | 0.989522219 |
| detail_positive_fraction | 60 / 20 | 0.844474637 | 1 | 0.3625 | 1 | 0 | 1 |
| detail_median_cosine | 60 / 20 | 0.407162164 | 0.484791324 | -0.0479928203 | 0.707313365 | -0.471539348 | 0.749714732 |

Rank ties use deterministic average ranks. Twenty singleton banks have undefined/null Spearman; no constant multistate bank occurred. Directional summaries with no eligible rows are null. Predicted/target argmin ties choose earliest state_index, and regret is measured in standardized target units. Worst-bank ties use ascending bank_index.

Worst relative-Huber banks (diagnostic only):
- Bank300,image45728: relativeHuber4.5651459693908691, Spearman-0.6452173913043479, regret6.0469279289245605.
- Bank327,image46463: relativeHuber0.26410940289497375, Spearman0.9677700348432056, regret0.025717735290527344.
- Bank152,image41488: relativeHuber0.14006431400775909, Spearman-0.0361026533275337, regret0.80345350503921509.

All80per-bank value/rank/regret/legacy/detail records and worst10lists are committed. No bank was removed or used for subgroup rescue. Good typical ordering coexists with severe errors on individual banks; subtracting a fixed offset does not make all within-bank geometry correct.

## Provenance and verification

Authorization `797a811da171b1b6a5875059fb210db7e924334f`; exact source `ce0c54ba8738a2378c2bf2de3d9376732e02abdd`,229source/Gitblob bindings. Frozen C2 checkpoint SHA256 `df874a53aadd359c1363e5866179a251dbf5ca86ceb72988d53485873202d93c`; immutable C2 split/complete marker and every accepted input file bound and rehashed. All input tensors/head hashes equal before/after. Reconstructed learned-detail gradient hash exactly matches C2; no new Jacobian or reference gradient was generated.

Sole CPU replay `20260918-010007-ttie-t059d-audit`, release `20260918-010002-ttie-t059d-audit`, exit0; diagnostic runtime40.80506018304732s. Local2tests3.39s/server2tests2.73s PASS. Command: focused pytest, then `research_log/T059D/run.py --out <artifacts/T059D>`, then separate `research_log/T059D/verify.py --out <artifacts/T059D>` with one-thread CPU environment. Separate verifier reopened every saved artifact, recomputed every per-bank diagnostic and global relative loss from row tensors, independently checked classification, and rehashed immutable inputs. No execution failure or scientific rerun.

Raw archive 292267bytes SHA256 `f7d636023f46a7bf3698d6dfc158d7794ac105e585b2e65d979cc311e275daa0`, verified remotehome/F. Exact full receipts/access and row tensors are compressed without changing decompressed bytes; all export hashes checked in memory before GitHub delivery. Ddisk ENOSPC interrupted the local download; that partial delivery archive is not authoritative. GitHub API publication uses verified server export bytes, and full home/F copies remain intact.

Counters allzero: training_runs,optimizer_steps,new_source_image_opens,reference_gradient_recomputations,new_feature_forwards,target_domain_access,lolv2_access,official_test_access,inference_reference_leakage. The frozen observed source holdout is diagnostic only and cannot be reused here to train/tune a replacement. No PSNR/SSIM, real-domain rollout, historical C2 modification or PROJECT_STATE edit. Stop for research-lead review.
