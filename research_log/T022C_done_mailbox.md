

---

# DONE T022-C — experiment-complete

Engineering evidence commit: ee7f7413427b77f9c9c2d2fa30a413e5b559611a. Scientific source: 824f36d9a7614acffa88977c4d128d0aa7afc85b. PR44 https://github.com/word-ky/TTIE/pull/44 is ready for research-lead review; no self-merge.

# T022-C completion report

Status: experiment-complete. The single predeclared dark EV cap change passes both fixed validation criteria: mean paired PSNR +0.9566850797500581 dB (threshold +0.50), mean paired SSIM +0.05522546532482912 (required nondecrease).

| Configuration | PSNR mean | PSNR median | SSIM mean | SSIM median |
|---|---:|---:|---:|---:|
| Raw | 8.109722671657906 | 7.600161540858166 | 0.1600228434803769 | 0.13897726549987016 |
| Accepted T022-A | 9.272868945613347 | 8.680092886100695 | 0.27300601233646227 | 0.24263053936978307 |
| T022-C | 10.229554025363404 | 9.650373829782268 | 0.3282314776612914 | 0.3002515375673881 |

| C minus A | Mean | Median | p10 | p90 |
|---|---:|---:|---:|---:|
| PSNR | 0.9566850797500581 | 0.776774357996247 | -2.7078585684847665e-08 | 2.8401093099062256 |
| SSIM | 0.05522546532482912 | 0.05443024892462302 | -1.1034073483795456e-07 | 0.12835192151337407 |

## Implementation and fixed protocol

Source commit 824f36d9a7614acffa88977c4d128d0aa7afc85b; subsequent evaluation binding commit recorded in git history. PR https://github.com/word-ky/TTIE/pull/44. Base task main889fc33191268c787c1db9efbb26e3467d9e2ab1. Explicit DarkEV2Box changes only active dark-winner EV upper +0.5 to +2.0. Bright [-0.5,0], inactive identity and gamma [0.8,1.25] remain exact. Historical source byte equality and unchanged trajectory AST are recorded in T022C_source_proof.json and focused tests. Frozen gate, head, CLIP, prototypes, identity initialization, Adam lr0.03, 40 updates, and minimum predicted-energy selection are unchanged.

The existing float32 renderer 2*tanh(raw) represents exactly +2 at finite raw10. The existing inverse bound mapping yields +inf at +2, making that raw upper clamp nonbinding while the physical renderer still caps EV at2. CPU and A6000 projection/endpoint and finite-gradient checks pass. No approximate endpoint or renderer change.

Same100 native600x400 validation inputs, split SHA256 b88c8347005984b5523b117b52c0c068672fe172eb7c9aa5b60102d350e2d85b. Each real image ran once on A6000 CUDA device1. No official-test decoding, scoring, sweep, retraining, or additional variant. Saved selected tensors are cloned without value changes, reducing output storage to288167700bytes by eliminating retained full-trajectory backing storage.

## Information boundary and verification

All100 low-only inference outputs/decisions/trajectories frozen at 2026-09-13T20:33:44.702006+00:00. Freeze SHA256 21173f3fb1d875d54ac93b8edf892391e3e9668a9a8bd63d2b06df7dec211423. New C evaluation references deployed only afterward at 2026-09-13T20:34:52.317835+00:00; evaluation completed20:35:22.552526+00:00. The inference runner has no normal-root argument and logs whitelisted low-only decoder access. Static source/trajectory tests and a dynamic counterfactual with100 synthetic normal files (unchanged, changed, withheld) preserve all inference artifact hashes; actual validation inference was never rerun for this test.

All100 gates equal acceptedA, all100 action bounds differ only at the declared dark upper cap, and all frozen scientific artifact hashes remain unchanged after reference evaluation. Raw metrics matchA exactly. Original accepted T022-A evaluator reused unchanged, full-frame float32 RGB inputs with float64 PSNR/accepted Gaussian RGB SSIM; separate Torch PSNR and explicit separable Gaussian SSIM audit maximum error5.329070518200751e-15. Independent NumPy/statistics aggregation and local CSV reaggregation agree. Six historical baseline tests and three focused local tests pass; the three focused tests also pass remotely, including the GPU endpoint check.

## Diagnostics

All100 runs use40 updates. Nonzero selected-step counts:26:1,33:1,34:2,35:4,36:1,37:1,38:1,39:1,40:88. The complete0..40 histogram is in comparison.json. Runtime seconds mean2.431485087046749, median2.4487252024991903, p95 2.479048882916686.

Selected-coordinate saturation (400 coordinates per channel; either-bound is a union including collapsed inactive bounds): EV lower7.25%, upper4.50%, either9.75%; gamma lower71.75%, upper15.25%, either85.00%. Saturation alone is not a causal proof and +2 need not be reached in40 updates. Absolute restoration quality remains modest. This result supports retaining the wider action range as a validation candidate; it is not an official-test or SOTA claim. No follow-on tuning was launched.

## Reproduction and artifacts

Run20260914-042925-ttie-t022c-ev2; exact launch command/environment in its run.sh and config.json. Server root /home/wenchang/asdasdsad/wjq/TTIE. Tracked compact evidence research_log/remote_runs/20260914-042925-ttie-t022c-ev2 contains100 decisions/trajectories, freeze/config, per-image metrics, paired_deltas.csv, comparison.json, independent_audit.json, comparison_verification.json and logs. All original output.pt files remain on the server. Asset identities reuse T022A_assets.json; source differences, evaluation binding, pre-reference manifest, reference deployment, test logs, and local receipt are research_log/T022C_*.

Full execution backup /media/wenchang/F/wjq/TTIE/shared/t022c/T022C_execution.tar SHA256 aff9747406cdce92105e8852e64b08b8d9f0e6731954319c457e2eddccd8d755. Compact transfer SHA256 5f3031412c1304fef7c4ae7cd861737fc826416b1c6045611b40515ccbe8113b. Engineering changes await research-lead review in PR44. Stop here pending the next OPEN task.

materially positive
