
---

## 2026-09-13T11:39:44Z — T020-B DONE — development target positive (5/5)

PR37 https://github.com/word-ky/TTIE/pull/37 . Branch `codex/T020B-target-viability`; scientific source `7134383cffd3abd8fe71090b3498bba6f837cb94`; evidence `8b6aa0eb5a84f916f19aa5d92c02e2ef820c28e1`. Exactly the accepted40 development images /120 clean-dark-bright episodes. This is a reference-only target audit, not fresh qualification or learned prediction. No selector, normalizer or model is fitted/modified; no new image or TTT trajectory is generated.

| Group | H0 | H_delta | H* | H_delta/H0 | H*/H0 | Beneficial/equal/harmful |
|---|---:|---:|---:|---:|---:|---|
| nonspatial_pool | 0.021385082408475378 | 0.019265628350876796 | 0.019217849626572085 | 0.9008910034988411 | 0.89865679540031274 | 61/59/0 |
| clean | 0.00028852283139713109 | 0.00020819390465476317 | 0.00020819390465476317 | 0.72158554540246811 | 0.72158554540246811 | 2/38/0 |
| homogeneous_dark | 0.045241762069053948 | 0.041020679031498732 | 0.040937563660554586 | 0.9066994112406046 | 0.90486227300497879 | 26/14/0 |
| homogeneous_bright | 0.018624962324975059 | 0.016568012116476894 | 0.016507791314506904 | 0.889559496947819 | 0.88632615875796195 | 33/7/0 |

Literal vector **[true,true,true,true,true]**: four pooled/condition mean-safety clauses all satisfy `mean(H_delta)<=1.01mean(H0)`, and harmful count=0. No rounded relaxation. Overall61 beneficial/59 equal/0 harmful. Clean has2 beneficial both-axis moves and38 no-move/equal episodes. All harmful/combined-harmful case lists are empty.

| Group | No move/x-only/y-only/both | x center/lower/upper | y center/lower/upper | Canonical in exact oracle tie set |
|---|---|---|---|---|
| nonspatial_pool | 59/7/10/44 | 69/30/21 | 66/35/19 | 57/120 (0.475) |
| clean | 38/0/0/2 | 38/0/2 | 38/0/2 | 38/40 (0.95) |
| homogeneous_dark | 14/1/5/20 | 19/14/7 | 15/17/8 | 14/40 (0.35) |
| homogeneous_bright | 7/6/5/22 | 12/16/12 | 13/18/9 | 5/40 (0.125) |

The positive-H0 axis function is byte-equivalent in logic to accepted T019-A (tested against its original function at PR32 head91f750e871d2c133624bbe4972f3fb3086f25a6c), with fixed delta=.01. A gain exactly at1% qualifies. When moving, minimum MSE chooses lower before upper; center→lower→upper tie order is preserved because qualifying positive gain makes center strictly worse. The explicitly requested H0=0 rule keeps center on both axes and records null gains, with no epsilon. Combined `(bx,by)` is looked up directly after independent choices; no joint reoptimization or fallback. Nine-hard oracle statistics are descriptive only and do not affect target selection or acceptance.

The development set is proven identical to accepted T018/T019 via six original Git commit:path inputs in `T020B_source_inputs/origins.json`: T015 development manifest/cache config/cache manifest, accepted T016 config, original T018 development reference table, and original T019-A source. Manifest SHA77f405417324fc4e2d50ccd4606caa06725416c9b0f5182daac5dbdc067a4a09 and cache config/manifest match the accepted T016 origin hashes. All40 IDs are explicitly listed in the lock and verification receipt. Conditions remain accepted T014 clean/homogeneous_dark/homogeneous_bright, gains1/.45/1.55, fixed order and preprocessing. No T020-A fresh references, features, logits, per-row outcomes or harmful-row-specific rule/subset are used; only its aggregate broad-safety-negative conclusion motivated this fixed diagnostic.

All68 baseline scientific files remain unchanged. The original byte-pinned T014/T015 identity input, selected Region2 grid, trajectory and nested checkpoint receipt are reused; frozen gate q_joint/tau/scale and energy match the accepted pipeline. GPU1 only renders the nine tau0 hard states over(.4,.5,.6)^2. Every one of120 canonical renders matches the original cached selected output pixel-for-pixel and exactly matches its MSE. No CLIP or adaptation rerun is needed. New code is `ttie/nonspatial_target.py`, launcher, focused tests and independent `T020B_verify.py`, plus source/lock/evidence artifacts.

Candidate table SHA `0c8c9c0ae20f590db1ecfb39f82540824e0a6c52ce43b0ff09e1231ac4fafc79`; cache binding SHA `17bdf8e38105286960247deeef89c4cd91cddd266d4947e984ccfb897d9a345f`; frozen11:32:17.385010Z before target reporting. The compact120-row evaluation records all nine reference MSEs, five cross values, gains, labels, combined choice/H_delta and H*. The independent verifier reconstructs exact threshold/tie/zero handling, combined lookup, five booleans, every movement/outcome/label/oracle statistic, all six original origins,78 runtime/source files and table/config/cache/state hashes, without importing TTIE or rerendering. Verification completed 2026-09-13T11:33:26.558977+00:00.

Commands: `python -m pytest tests/test_nonspatial_target.py tests/test_nonspatial_safety.py -q` =>11passed22.86s; baseline11passed14.00s; repair-focused `python -m pytest tests/test_nonspatial_target.py -q` =>3passed15.51s. `bash scripts/run_t020b_a6000.sh 7134383cffd3abd8fe71090b3498bba6f837cb94`; independent `python research_log/T020B_verify.py --output research_log/remote_runs/20260913-193203-ttie-t020b-target-audit-fixed/artifacts/audit --receipt research_log/T020B_verification.json`. Completed run `20260913-193203-ttie-t020b-target-audit-fixed` exits0 on A6000 GPU1/Torch2.4+cu121, release `20260913-193127-ttie-t020b-ready`.

Observed engineering failures are retained: initialrun `20260913-192817-ttie-t020b-target-audit` (source7a10d5188144d5979256cacc6dd9ffe723f9da23) exited1 at row0 before any candidate result because cached `selected_step` is nested under `selection`. A one-line reader repair and focused tests precede the completed run; no scientific rule changed. An intermediate unlaunched deployment lacked Gitpack/index because staging had not completed; the completed stage was redeployed and passed all78 source/input checks before launch. A whole-gate-receipt comparison initially included extra provenance fields; actual numeric gating fields were equal. No scientific tuning or image replacement occurred.

Both the failed and completed run files are retained/mirrored:136 files492234bytes. All480 original cache files used here (1225337177bytes) are hash-matched home/F; missing home copies were restored from the intact validated F cache. `T020B_remote_mirror.json` records all hashes. Source, compact states/tables and recovery notes stay project-local.

Interpretation/recommendation: the fixed reference target is viable on these non-spatial development cases, supporting the target-safe explanation within this set and leaving learned movement generalization unresolved. This does not prove fresh target safety or that any revised predictor will succeed. T020-A stays fresh-negative; T014 stays the broad deployable baseline. Stop for research-lead review. No training, selector change, new cohort, T020-C or downstream experiment is running or authorized within this cycle. Research-owned PROJECT_STATE.md remains untouched.
