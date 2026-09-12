# Codex → ChatGPT

Codex: append implementation reports below. Do not overwrite previous reports.

Use the format defined in `coordination/PROTOCOL.md`.

---

## Awaiting report

Open task: `T001 — Minimal Spatial TTT-ISP Mechanism Scaffold`

## T001 — DONE (implementation; research acceptance pending)

UTC: 2026-09-11T17:06:00Z

- PR: https://github.com/word-ky/TTIE/pull/1
- Branch: `codex/T001-spatial-isp-scaffold`.
- Tested code SHA: `17f6563f5d6aa0532aa8fab3cc6023d6f7195b4b`.
- Evidence SHA: `99a1cc2829d241b0eebf2ee298568aabc56acadd` (full A6000 receipts and recovery notes; no research-code changes).
- Source baseline: `367e3c33c28a1781d03b72aec4734be032e78b7f`, documents only. No donor project code copied.

### Implementation and files

`ttie/isp.py`: exposure, shifted gamma, RGB WB and contrast, shared by global 1x1 and spatial configurable bilinear grids. Physical EV bounds [-2,2], other coordinates [0.5,2], identity at raw zero. Six global versus 96 spatial parameters (4x4).

`ttie/adapt.py`: image + label-free loss callable API, fresh identity and Adam state for each episode, only raw ISP gradients/updates, returned image/raw grid/physical grid/full field and all step losses, gradient norms, coordinate ranges and finiteness.

`ttie/demo.py`: deterministic 64x96 RGB toy, left x0.45/right x1.55 with clipping, identity/global/spatial evaluation, full JSON/float tensor/PNG outputs. Fixed prior is mean((avg_pool2d(output,8)-0.5)^2), not derived from a clean target. Both adaptations complete before reference evaluation. No target/label argument in adaptation; the reference-replacement test changes evaluation MSE without changing adapted outputs or loss trajectories.

`tests/test_isp.py`, `tests/test_adapt.py`, `tests/test_demo.py`: 13 CPU tests. `scripts/run_a6000.sh`: remote CPU suite and CPU/CUDA/repeat experiment. `README.md`, `requirements.txt`, `.gitignore`, `.gitattributes`: usage, conventions, minimal dependencies and shell LF policy. `research_log/`: working log, remote recovery instructions, exact test output and full experiment receipts/artifacts. No detector, CLIP, free pixel field, meta-learning, external dataset or future-task module added.

### Commands and exact outcomes

From project root:

```text
python -m unittest discover -s tests -p test_isp.py -v
Ran 5 tests in 3.027s — OK
python -m unittest discover -s tests -v  [after adaptation increment]
Ran 10 tests in 12.812s — OK
python -m ttie.demo
exit 0, fixed default seed 7 / 200 steps / lr 0.03 / 4x4 grid
python -m unittest discover -s tests -v  [full local suite]
Ran 13 tests in 14.042s — OK
git diff --check
passed
```

Remote workflow root `D:/work/claude-autodl/autodl-workflow-clean`, with `AUTODL_CONFIG_PATH` set to TTIE's ignored `.autodl/config.json`:

```text
scripts/autodl-deploy.ps1 -Tag ttie-t001 -Source D:/work/fightccfa-agin/CVPR2027/TTT-ImageEnhancement
release 20260912-010250-ttie-t001
scripts/autodl-run.ps1 -Name ttie-t001-a6000 -Cmd 'bash scripts/run_a6000.sh'
run 20260912-010311-ttie-t001-a6000
remote CPU suite: Ran 13 tests in 3.060s — OK
CPU demo + CUDA demo + CUDA repeat: exit 0
CUDA repeated reports and tensors match exactly; all finite.
run finished 2026-09-11T17:03:45Z, exit_code=0
```

Local: Python 3.12.7, torch 2.13.0+cpu, Pillow 10.4.0. A6000: Python 3.12.12, torch 2.4.0+cu121, CUDA 12.1, Pillow 12.3.0. No local GPU available; CUDA was tested on the actual RTX A6000. Automated tests cover identity (including black/white), each operator's analytic value and finite nonzero gradient, spatial interpolation, global/constant-field equality, fast-only updates, episode reset, CPU, analytic metrics, deterministic full demo and clean-reference independence.

### Measured results (A6000 CUDA)

| Case | Self-supervised loss before | Loss after | MSE to clean (evaluation only) | PSNR dB |
| --- | ---: | ---: | ---: | ---: |
| Identity | 0.07551638 | 0.07551638 | 0.07748476415872574 | 11.1078 |
| Global | 0.07551638 | 0.00623847 | 0.00988832302391529 | 20.0488 |
| Spatial 4x4 | 0.07551638 | 0.00078186 | 0.002459792885929346 | 26.0910 |

Spatial MSE is **75.12426647086426% lower** than global (+6.0422 dB). Left/right MSE: global 0.01418523/0.00559142, spatial 0.00243114/0.00248844. No NaN/Inf occurred in outputs, losses, gradients or physical fields. CPU versus CUDA spatial MSE differs by approximately 4.7e-10; duplicate CUDA tensors and reports are bitwise identical.

Learned global [EV, gamma, WB-R, WB-G, WB-B, contrast]: [-0.520365, 0.520844, 0.827869, 0.827636, 0.827754, 0.530883]. Spatial coordinate minima: [-0.789777, 0.592122, 0.738297, 0.738223, 0.738352, 0.566599]; maxima: [0.581452, 0.942055, 1.187374, 1.187655, 1.188322, 0.725919]. Full 4x4 physical grids and every step's ranges/gradient norm/loss are in JSON; full-resolution fields and raw parameters are in tensors.pt.

Artifacts at evidence SHA/PR branch:

- `research_log/artifacts/T001_cpu/`: original local metrics.json, tensors.pt, comparison.png.
- `research_log/remote_runs/20260912-010311-ttie-t001-a6000/`: train.log, meta.json, run.sh; artifacts/environment.txt, tests.txt, reproducibility.txt; artifacts/cpu, cuda and cuda_repeat each containing metrics.json, tensors.pt and comparison.png.
- `research_log/T001.md`, `research_log/REMOTE.md`: durable implementation and remote recovery records.

### Interpretation, deviations and failures

The specified controlled toy experiment supports the global-vs-spatial recovery hypothesis. It does **not** establish general natural-image restoration or downstream-task utility. The toy is centered near the fixed 0.5 statistical prior; this favorable match is explicit. That prior can flatten texture and real color and does not distinguish dark objects from exposure. Global output visibly loses contrast; spatial also has residual transition/texture error. 0.8681% of degraded pixels are clipped. Four operator roles are non-identifiable because exposure/WB/gamma/contrast can compensate. Final fixed-step outputs were evaluated; no clean-metric hyperparameter or checkpoint selection was performed.

No scientific scope deviation. Concrete choices within T001: default 4x4 physical grid, shifted power for finite black-pixel gamma gradients, Adam, fixed patch RGB-mean prior, direct fast-grid optimization (no token encoder/fast network).

Observed setup failures resolved: default remote shell lacked python (used verified interpreter and TTIE-local venv); Pillow absent in that venv (installed there); git author identity missing (set repository-local Codex identity). No failed scientific run or modified evaluation target. An existing environment protobuf deprecation warning did not affect tests. No external datasets, real detector experiments or broader benchmark validation were run, as excluded by T001.

### Communication and next step

Per user's additional instruction, created project `/home/wenchang/asdasdsad/wjq/TTIE` on the A6000 under the existing writable wjq directory. No TTIE job remains running. Code current release and run receipts remain there; full results are also local and in this PR.

Created ACTIVE Codex task heartbeat `ttie-chatgpt`, every 15 minutes. It reads main inbox/state and PR review feedback, continues authorized work, writes the Codex mailbox, avoids duplicate execution of reported tasks still marked OPEN, and stays quiet when unchanged. The user's ChatGPT hourly heartbeat is reported, not independently verified; the first future scheduled two-way exchange remains to be observed. GitHub is the synchronization channel, not a direct ChatGPT process-control API.

Recommended next step: research lead reviews PR #1, accepts or requests revisions, and publishes the next scoped task. Before interpreting broader recovery gains, consider a task on spatial parameterization with suitable texture-preservation and illumination/content-confound controls. These are suggestions only; no T002 or future modules implemented. Research lead retains ownership of task status and PROJECT_STATE.


## T002 — PARTIAL (implementation tested, A6000 matrix running)

UTC: 2026-09-11T18:03:00Z

The Codex 15-minute heartbeat received the T001 acceptance and OPEN T002 at 17:54:54Z; PR #1 was already merged. GitHub-mediated two-way communication is now observed. Code branch `codex/T002-spatiality-controls`, SHA `052a4361df983da812213c853459b1cb84f4558c`, has the uniform latent-count control, synthetic suite, tests and predeclared experiment in research_log/T002.md. Full local CPU regression: 20 tests passed in 5.089s. T001 APIs/semantics retained; 1x1/global outputs equal exactly in tests.

Predeclared matrix: seeds 7/11/23, three content families, eight conditions including clean/homogeneous/conflicting/smooth/high-frequency shifts, seven variants = 504 rows. Same 200-step Adam, lr 0.03 and unchanged label-free objective throughout. Uniform96 averages the 16 raw six-vectors before bounds: 96 optimizer scalars but six effective output degrees of freedom, explicitly a latent-count rather than functional-capacity control.

A6000 release `20260912-020216-ttie-t002`, run `20260912-020232-ttie-t002-a6000`, command `bash scripts/run_t002_a6000.sh`, project `/home/wenchang/asdasdsad/wjq/TTIE`. CPU suite and T001 regression, all-variant CPU/CUDA parity and exact CUDA repeat checks precede the full CUDA matrix. No outcome claims yet; final metrics/figures and hypothesis verdict will follow when the run completes. No current blocker.

## T002 — DONE (implementation/evidence; research acceptance pending)

UTC: 2026-09-11T18:27:00Z

- PR: https://github.com/word-ky/TTIE/pull/2
- Branch: `codex/T002-spatiality-controls`.
- Tested source SHA: `052a4361df983da812213c853459b1cb84f4558c`.
- Evidence SHA: `b248fd27a11f99441a7e823c2af30569bbb64fdf`. Subsequent code/tests/scripts diff from tested source is empty; only communication/evidence changed.
- Base: accepted T001 on main `54362b1f661777d26d56592f4594323ee5f944a5`. Research-owned inbox/state unchanged by Codex.

### Files and implementation

`ttie/isp.py` adds uniform_control: average 16 raw six-vectors before the unchanged bounded ISP mapping and broadcast the single physical vector everywhere. Latents start at zero symmetrically; 96 scalars are optimized, but the rendered function still has only six effective degrees of freedom. `ttie/adapt.py` returns the rendered physical grid through this shared ISP interface; global/spatial defaults and no-label boundary are unchanged.

`ttie/suite.py` generates three clean families and eight illumination conditions, runs identity/global/uniform96/spatial1/2/4/8, and exports complete metrics, fixed representative tensors/diagnostics and RGB/EV-field figures. `ttie/summarize.py` aggregates all seeds without selection. `tests/test_uniform.py`, `tests/test_suite.py` add seven focused tests; prior tests remain. `scripts/run_t002_a6000.sh`, README and research_log/T002.md document/run the protocol. Full receipts, local test output, preview and final figures are under research_log.

Predeclared fixed matrix: 3 seeds (7/11/23) × 3 families (midtone, intrinsic dark structures, high-key) × 8 conditions (clean, homogeneous dark/bright, left/right, quadrants, smooth gradient, four/twelve stripe cycles) × 7 models = **504 rows / 432 adaptation episodes**. Adam 200 steps, lr 0.03, same zero initialization, operator bounds and T001 8x8 patch-mean-to-0.5 prior everywhere. No clean-reference tuning or model/checkpoint selection. Input-only adapt_variants receives no clean image, condition, masks or gains; clean-reference evaluation happens after all variants finish adapting that input.

### Commands / validation

Local project root:

```text
python -m unittest discover -s tests -v  [baseline T001]
13 passed in 5.698s
python -m unittest discover -s tests -p test_uniform.py -v
3 passed in 3.139s
python -m unittest discover -s tests -p test_suite.py -v
4 passed in 2.986s
python -m unittest discover -s tests -v  [complete local suite]
20 passed in 5.089s
git diff --check
passed
```

Existing AutoDL workflow root, using TTIE's .autodl/config.json through AUTODL_CONFIG_PATH:

```text
scripts/autodl-deploy.ps1 -Tag ttie-t002 -Source D:/work/fightccfa-agin/CVPR2027/TTT-ImageEnhancement
release 20260912-020216-ttie-t002
scripts/autodl-run.ps1 -Name ttie-t002-a6000 -Cmd 'bash scripts/run_t002_a6000.sh'
run 20260912-020232-ttie-t002-a6000
remote CPU full suite: 20 passed in 3.152s
T001 fixed numeric regression: passed, MSE tolerance 1e-9
seven-variant CPU/CUDA parity: passed, tensor atol 1e-5 / rtol 1e-4
CUDA repeated rows, diagnostics and tensors: exact match
python -m ttie.suite --device cuda:0 --output "$AUTODL_ARTIFACTS_DIR/suite"
python -m ttie.summarize "$AUTODL_ARTIFACTS_DIR/suite"
504 unique rows; all finite; exit_code=0 at 2026-09-11T18:23:14Z
```

Local Python 3.12.7 / torch 2.13.0+cpu / Pillow 10.4.0. Remote Python 3.12.12 / torch 2.4.0+cu121 / CUDA 12.1 / Pillow 12.3.0 on RTX A6000. The full matrix was executed once on actual CUDA; CPU validation and fixed parity/repeat cases are separate from the matrix. No mock-only result presented as a real experiment.

### Results and hypothesis verdicts

Uniform96 physical field variance is **exactly zero** across all inputs. Spatial1/global MSE matches exactly for all 72 inputs. Max |uniform96-global| MSE is **9.84582584e-6**: the symmetric 1/16 raw-gradient scaling changes Adam epsilon's relative influence slightly, as documented before execution.

Mean evaluation MSE over all three noise seeds, midtone content:

| Shift | Uniform96 | Spatial2 | Spatial4 | Spatial8 | Spatial4 reduction vs uniform96 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Homogeneous dark | 0.0011337 | 0.0011337 | 0.0011337 | 0.0011339 | -0.002% |
| Homogeneous bright | 0.0003329 | 0.0003330 | 0.0003331 | 0.0003335 | -0.066% |
| Left/right | 0.0098803 | 0.0042488 | 0.0024584 | 0.0015818 | 75.118% |
| Quadrants | 0.0098743 | 0.0056981 | 0.0040560 | 0.0038780 | 58.924% |
| Smooth gradient | 0.0058404 | 0.0022995 | 0.0010770 | 0.0007939 | 81.560% |
| Four stripe cycles | 0.0122895 | 0.0121535 | 0.0120531 | 0.0115963 | 1.923% |
| Twelve stripe cycles | 0.0504809 | 0.0513388 | 0.0520596 | 0.0535242 | -3.127% |

- **H1 conditional:** homogeneous global/spatial performance is nearly equal for midtone/high-key content. On intrinsic dark structures, spatial4 is 19.408% worse for homogeneous darkening and 12.994% worse for homogeneous brightening. No universal equivalence claim.
- **H2 conditional:** spatial fields substantially help resolvable midtone left/right, quadrant and gradient shifts. Higher grid resolution helps there; gains collapse or reverse on dense stripes. Dark-content left/right spatial4 is 7.784% worse than uniform96. Eight-by-eight loss averaging/aliasing, field resolution, content prior and clipping are entangled; the suite does not isolate a pure representation-frequency cause.
- **H3 narrow support:** gains persist against the requested many-latent uniform control, ruling out mere redundant optimizer scalar count. Averaging still leaves only six effective output degrees of freedom, so the control does **not** eliminate effective function-capacity confounding. It is not evidence against a richer globally uniform nonlinear ISP.
- **H4 clearly supported:** even undegraded images are changed while self-supervised loss decreases. Dark objects brighten; high-key content darkens; texture is suppressed. More spatial freedom can amplify damage.

Mean clean/no-degradation identity-drift MSE (identity baseline is zero):

| Clean family | Global | Uniform96 | Spatial4 | Spatial8 |
| --- | ---: | ---: | ---: | ---: |
| Midtone | 0.0005575 | 0.0005491 | 0.0005393 | 0.0004748 |
| Dark structures | 0.0176932 | 0.0176932 | 0.0203560 | 0.0262865 |
| High-key | 0.0628824 | 0.0628824 | 0.0628824 | 0.0628825 |

High-key smooth-gradient spatial4 improves over uniform96 but is still worse than no adaptation (MSE 0.0634104 versus 0.0478751). This is why the identity baseline remains essential. Input clipping is 100% on high-key homogeneous brightening and 50% on its left/right/quadrant/stripe cases. Every row includes its actual input/output clipping; maximum adapted-output clipping is 0.016276%. All output/loss/gradient/physical fields are finite. Full-image versus mean quadrant MSE differs by at most 3.35e-8 due to float reductions. Exact-zero identity/clean MSE has infinite PSNR encoded as JSON null / CSV blank.

### Artifacts, failures, deviations and next action

Complete results at the PR/evidence SHA: `research_log/remote_runs/20260912-020232-ttie-t002-a6000/`. The run contains meta.json, run.sh, train.log, environment/tests/parity text; artifacts/suite contains config.json, compact 504-row metrics.csv/metrics.json, summary.json/summary.md, 10 fixed seed-7 tensor packs and full trajectories, `degradations.png` and `content_drift.png`. Figures were visually checked. `research_log/T002.md` gives the full interpretation and provenance. Remote receipts remain at `/home/wenchang/asdasdsad/wjq/TTIE/runs/<id>`; project recovery logs are mirrored under its research_log. No TTIE job remains running.

No runtime/scientific execution failures or blockers, no scientific scope deviation, no configuration tuning and no code change after tested source. Negative results above are measured method failures, retained in full. Three seeds vary synthetic noise only, not independent natural scenes; no natural-image, detector or statistical-generalization claim. No CLIP, detector, new loss, identity anchor, edge-aware module, meta-learning, ViT3 or T003 implemented.

Recommend research-lead review of PR #2 and an objective/content-preservation diagnostic before adding downstream models. A strict effective-capacity claim would require another appropriately matched baseline beyond uniform96. Codex awaits acceptance/revisions/the next scoped task; the 15-minute mailbox heartbeat remains active. T001 acceptance and the T002 reply were received through an actual scheduled GitHub round trip.

## T003 — PARTIAL (implementation tested; fixed A6000 sweep running)

UTC: 2026-09-11T19:13:00Z

Received T002 acceptance / PR #2 merge and OPEN T003 via the 19:00:25Z heartbeat. Branch `codex/T003-objective-safety`, tested source `7469990f491a2295f86b4095a07bd794a47a28e7`. Normalized physical-grid anchor and plain TV are implemented, with prior/anchor/TV/total trajectories and unchanged zero-weight behavior. All 27 local CPU tests passed in 14.915s, including the prior regressions, global TV invariance, clean-reference replacement and known-number decision-rule test.

Predeclared exact 11 weight settings, all T002 seeds/families/conditions, fixed Adam 200/lr .03, identity/global references: 936 rows / 864 reset episodes. Full scan uses CPU on the A6000 host; separate CUDA exact-repeat and cross-device sensitivity measurements precede it. This compute choice was made before outcomes because the prior tiny-image CUDA sweep was synchronization-heavy. The 5x worst-drift / 70% utility conjunction is unchanged. Worst means maximum over all nine clean inputs; worst family mean is additionally reported. Representatives are fixed before outcomes at (0,0), (.1,.1), (1,.1), (10,0) for the three requested cases. No per-image hyperparameter selection.

Release `20260912-031235-ttie-t003`, run `20260912-031249-ttie-t003-a6000`, command `bash scripts/run_t003_a6000.sh`, project `/home/wenchang/asdasdsad/wjq/TTIE`. Read research_log/T003.md on the branch for full predeclaration. All trajectories are saved for every input/setting. Matplotlib is available locally and Pareto figures will be rendered from final metrics locally; remote execution uses existing torch/Pillow. Final verdict pending completion; no method outcome claimed yet.

## T003 — DONE (negative fixed diagnostic; final acceptance pending)

UTC: 2026-09-11T19:34:00Z

PR: https://github.com/word-ky/TTIE/pull/3
Experiment source: 7469990f491a2295f86b4095a07bd794a47a28e7.
Reporting-only repair: 1e7d38b8a7d631341d45a4d4fcf516698aa9b685.
Complete evidence: 1d416f839753df3bee6515d462e78347f3c62aec.

Received interim research-lead review a6ced5015c3ff777a0c66a96c40ee9618dda54af. Its implementation acceptance, precise device scope and freeze instructions are respected: no experiment configuration/code, aggregation formulas, representatives, weights or optimizer changed after outcomes. No additional sweep or module implemented. The original run finished successfully before this report and PR.

### Implementation and validation

Added normalized physical-grid anchor and plain TV, decomposed prior/anchor/TV/total trajectories, fixed 11-setting sweep, exact decision summary and all-point figures. Existing ISP, label-free 0.5 prior and zero-weight behavior preserved; clean target remains evaluation-only. Full matrix: 72 identical T002 inputs x 13 variants = 936 unique finite rows, 864 reset episodes, 72 complete trace files. Each adapted trace has 201 component/total samples and 200 gradient samples. All seeds/families/conditions, Adam lr .03 and 200 steps match the task.

Commands: python -m unittest discover -s tests -v; bash scripts/run_t003_a6000.sh; python -m ttie.safety_sweep --device cpu --output <artifacts>/sweep; python -m ttie.safety_summary <artifacts>/sweep; local figure generation adds --plot to the summary command.

Local baseline 20 tests passed; initial implementation 27 passed in 14.915s; remote 27 passed in 4.998s; final reporting-fix 27 passed in 8.095s. Tests include identity/finite normalization, analytic penalties, global zero TV, complete trajectory clean-reference replacement, frozen T001 numeric regression and known-number decision arithmetic. git diff --check passed.

Release 20260912-031235-ttie-t003, run 20260912-031249-ttie-t003-a6000, completed 2026-09-11T19:23:26Z with exit 0. This is an **A6000-host CPU sweep plus focused CUDA validation**. Remote Python 3.12.12 / torch 2.4.0+cu121 / CUDA 12.1; local Python 3.12.7 / torch 2.13.0+cpu. All 13 CUDA repeated outputs/rows/diagnostics match exactly. CPU/CUDA TV sensitivity reaches max MSE difference .0001512579619884491 and max output difference .011940419673919678. These are reported differences, not exact parity. The complete matrix consistently uses CPU, predeclared before outcomes and accepted in interim review.

### Complete fixed decision table

936 rows; all finite: True.

One setting must reduce worst clean drift by at least 5x AND retain at least 70% of baseline improvement. These are diagnostic thresholds, not a per-image selection policy.

Baseline worst drift: 0.0628858656; safety ceiling: 0.0125771731; baseline mean MSE improvement: 0.00600123236.

| Setting (anchor, TV) | Mean clean drift | Worst clean drift | Worst family mean | Drift reduction | Utility retained | Meets both |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| (0, 0) | 0.0279259 | 0.0628859 | 0.0628824 | 1.000x | 100.000% | False |
| (0.01, 0) | 0.0267567 | 0.0622966 | 0.0622932 | 1.009x | 104.931% | False |
| (0.1, 0) | 0.0233593 | 0.0599426 | 0.0599393 | 1.049x | 78.810% | False |
| (1, 0) | 0.0153735 | 0.0430007 | 0.0429983 | 1.462x | -54.356% | False |
| (10, 0) | 0.0030384 | 0.0089490 | 0.0089483 | 7.027x | -516.828% | False |
| (0, 0.01) | 0.0273664 | 0.0628816 | 0.0628802 | 1.000x | 92.302% | False |
| (0, 0.1) | 0.0257677 | 0.0628020 | 0.0627932 | 1.001x | 72.599% | False |
| (0, 1) | 0.0226825 | 0.0629759 | 0.0629259 | 0.999x | -192.740% | False |
| (0.1, 0.1) | 0.0221615 | 0.0600628 | 0.0600088 | 1.047x | 53.862% | False |
| (1, 0.1) | 0.0150028 | 0.0429802 | 0.0429731 | 1.463x | -106.000% | False |
| (1, 1) | 0.0149865 | 0.0432249 | 0.0431449 | 1.455x | -403.836% | False |

Qualifying fixed settings: NONE

Safety uses the worst individual input among the nine clean cases; the alternative worst family-mean is displayed without changing the decision rule. Utility is a ratio of mean improvements, not a mean of per-image ratios. Negative utility is retained, not clipped.

All input-level errors, region errors, clipping and grid diagnostics remain in metrics.csv/metrics.json. All prior/anchor/TV/total and gradient trajectories are in trajectories/. Seeds vary synthetic noise only. Passing does not establish natural-image identity safety; failing rules out only this finite diagnostic weight sweep and optimization budget.

### Interpretation, observed failures and limits

**No setting qualifies. Simple identity anchoring/plain smoothness does not rescue the absolute 0.5 prior under this diagnostic.** Strong anchor (10,0) meets safety at 7.027x lower worst drift but loses utility (-516.828% retention means worse than global aggregate, not negative MSE). Settings retaining >=70% utility improve worst drift no more than 1.049x. No near miss is reclassified or per-case weight selected.

TV optimization is a material limitation: 38/864 episodes finish with total loss > initialization + 1e-8, all TV settings. Counts: (0,.01):3; (0,.1):3; (0,1):6; (.1,.1):3; (1,.1):5; (1,1):18. Mean final-20 total-loss standard deviation reaches .000683922 for (0,1) and .000678854 for (1,1). No anchor-only episode has an increase; strong anchor (10,0) has stable final loss (mean std 1.06256e-08) yet fails utility. Do not claim all settings converged or all possible regularizers must fail. The result covers these 11 weights and this optimization budget, with three noise seeds on synthetic content.

Maximum loss-component reconstruction error is 1.024454832e-08. Unregularized global/spatial4 MSE differences from T002 CUDA are <=9.685754776e-08 / 4.470348358e-08. Maximum adapted output clipping is .001030815998 of RGB samples (about .1031%); all input clipping/error/region/field diagnostics are retained.

The sole runtime failure was local figure rendering: Windows OMP Error #15 from needless Torch imports in safety_summary. Reporting-only fix 1e7d38b derives metadata from completed rows and imports matplotlib lazily; no runtime-bypass flag. Recomputed JSON equals original remote summary_remote.json exactly. Experimental source files remain unchanged from 7469990. Pareto PNG/SVG and the fixed three-case panel now render and have been visually inspected. No failed experiment or hidden rerun.

### Artifacts and next action

All raw receipts: research_log/remote_runs/20260912-031249-ttie-t003-a6000/ at evidence SHA/PR. Includes meta/run/log, environment/tests/device checks; artifacts/sweep contains config, all 936 metrics.csv/json rows, 72 trajectory JSONs, three representative tensor packs, original/final summaries, audit_summary.json, content_utility_panel.png and pareto.png/svg. research_log/T003.md contains full provenance, interpretation and manifest; local final test receipt is research_log/T003_tests_cpu_final.txt. Remote run remains under /home/wenchang/asdasdsad/wjq/TTIE/runs/<run-id>; recovery notes mirrored to the remote project root research_log.

No TTIE job remains active. Recommend research-lead acceptance/review of PR #3 and specification of the semantic/degradation-aware next task as stated in the interim instructions. No extra regularization weights, edge-aware term, teacher, CLIP, detector, new dataset or T004 implemented. The 15-minute heartbeat remains active and will await actionable feedback; completed T003 will not be rerun solely because the inbox still says OPEN.

## T004 — PARTIAL (Stage-A protocol implemented; checkpoint transfer in progress)

UTC: 2026-09-11T20:50:00Z

Received T003 acceptance/PR #3 merge and OPEN T004 through the 20:39:27Z heartbeat. Source/protocol commit 28c696407724bdc17631b6503a8eba4df81830b1 on codex/T004-clip-signal. No held-out CLIP scores or adaptation outcomes have been inspected. All 34 local tests passed in12.949s, including27 prior regressions and7 new data/calibration/gradient-ownership/audit tests. Real pretrained CLIP CUDA validation remains pending; mock contract tests are not experimental evidence.

Fixed preferred model ViT-B-32/laion2b_s34b_b79k, exact nine prompts/five crops, clean calibration95th percentile and population std floor .01, unchanged Stage-A gate. Data manifest committed before scoring: metadata-sorted first18 eligible COCO files from an existing200-image val2017 cache (not the first18 of all5000); first6 calibration, next12 evaluation. Original shorter side>=320; IDs, dimensions, URLs and SHA256 in research_log/T004_manifest.json. No annotations or other project results read. Selection pool provenance is limited and will be disclosed; no substitute dataset or score-based selection. Files copied into TTIE/shared/t004/images. RGB source preprocessing shorter side320, CLIP views shorter side224/center224, bicubic antialias with official mean/std.

Installed open_clip_torch2.26.1 and torchvision0.19.0 in TTIE venv, preserving torch2.4.0+cu121. Concrete setup failure: A6000 direct Hugging Face access returns Errno101; Windows can access the exact same official605219813-byte checkpoint. Download locally then checksum/transfer to TTIE shared storage; model unchanged. research_log/T004.md records all predeclarations, commands and tests. No PR before final Stage-A verdict; Stage B will run only if the literal fixed gate passes, otherwise stop and report. No T005.

## T004 — DONE (Stage-A gate failed; Stage B not authorized/run)

UTC: 2026-09-11T20:59:00Z

PR: https://github.com/word-ky/TTIE/pull/4
Branch: codex/T004-clip-signal.
Pre-score protocol/manifest commit:28c696407724bdc17631b6503a8eba4df81830b1.
Checkpoint identity committed before scoring:1dcda3b31c782de76e339ff52f7a0702bd78b729.
Successful tested source:ddcafb06c530f486c49ca3b779e9640d454172ee.
Complete evidence:832e1cdd420cf501552c3089f49e96dc8f294662.

**The literal Stage-A gate fails because bright-vs-clean ROC-AUC is0.598611<0.75. T004 stops here.** No evaluation adaptation, EV/Gamma mask implementation, reconstruction result, Stage-B panel, prompt/model/threshold tuning or T005 was run. This is a completed negative signal audit, not a blocked experiment.

### Scope, provenance and implementation

Used prescribed frozen OpenCLIP ViT-B-32/laion2b_s34b_b79k and all nine exact text prompts, normalized individual embeddings then average/renormalize each concept. Weights eval/frozen; all adaptation-relevant scoring APIs take only pixels and fixed calibration. New files ttie/natural.py, clip_signal.py and clip_audit.py provide natural-image geometry, fixed views, scores/calibration/gating and offline audit. Existing T001–T003 ISP/adapt implementations unchanged. Tests added in test_natural.py/test_clip_signal.py; remote entry scripts/run_t004_a6000.sh; dependencies requirements-clip.txt. A reusable exact inactive-zero semantic-loss primitive is tested, but no real Stage-B episode exists.

Before any CLIP score, metadata-only18-image manifest committed at research_log/T004_manifest.json. COCO2017-val source was the existing200-JPEG image-only cache /home/liujianhua/wjq/TAISP/shared/coco200/val2017. Sorted available IDs, original minimum side>=320; first6 calibration, next12 held out. This is not the first18 of all5000 images; the initial200-image pool's selection provenance is incompletely known. No annotations or other project scores/results read. Source URLs/IDs/dimensions/SHA256 and excluded short image recorded; selected files copied into TTIE/shared/t004/images, no image data committed. Pilot sampling limitation is explicit.

RGB float32 whole-image shorter-side320, aspect preserved, bicubic antialias/clamp. Six exact task exposure conditions reuse previous gain conventions. Five fixed full/quadrant views use differentiable shorter-side224/center224 with official normalization. Odd-size torchvision tensor geometry matches exactly. Metadata is attached only after scoring; masks/labels do not enter the scorer. All mixed-condition scores and per-view clipping are retained without assigning a single true type to mixed full-image views.

Official checkpoint605219813bytes, SHA2561bd3c7172de5b207ceac554f5ab5266166f3b9baccc9af5989bc801016d080ad. Identity in research_log/T004_model_identity.json. Remote Python3.12.12, torch2.4.0+cu121 unchanged; torchvision0.19.0/open_clip_torch2.26.1 installed only in TTIE venv. Frozen clean calibration constants (dark,bright): tau=(.026910689473152147,.029729080200195306), scale=(.032711278852056797,.02081351470293608), linear95th percentile/population std floor.01 on30views from six calibration IDs only. Persisted before held-out scoring; no constants refitted on evaluation.

### Commands, tests and actual A6000 evidence

Local: python -m unittest discover -s tests -v. Baseline27 passed in11.419s; natural2tests passed in1.623s; signal5tests passed in1.589s; full initial34 passed in12.949s; final34 passed in10.096s. Covers all27 old regressions plus gradient ownership, differentiable views/legacy gains, calibration exclusion of held-out and degraded calibration rows, detached winner/strict threshold, inactive zero objective, pixel-only score independence and tied-AUC/literal gate. Mock encoder tests are unit evidence only.

Remote: bash scripts/run_t004_a6000.sh; runs full34tests (passed in5.282s), then python -m ttie.clip_audit --manifest research_log/T004_manifest.json --images /home/wenchang/asdasdsad/wjq/TTIE/shared/t004/images --model-identity /home/wenchang/asdasdsad/wjq/TTIE/shared/t004/model_identity.json --output <artifacts>/audit --device cuda:0. Release20260912-045534-ttie-t004, run20260912-045537-ttie-t004-a6000, exit0 at20:55:57Z. Full Stage A actually ran on RTX A6000 CUDA in float32. All390unique rows finite:30calibration +360held-out (12images x6conditions x5views). Held-out scoring executed once.

Actual pretrained CLIP gradient check on one calibration image: every model parameter frozen/no accumulated gradients, finite nonzero input-pixel gradient norm .07762914150953293. Repeated CUDA gradient max difference3.4924596548080444e-10; gradient/no-grad forward-score difference7.450580596923828e-08. Do not claim exact bitwise repeats. All Stage-B-only adaptation/reference-replacement/identity/reset/mask tests remain unimplemented/unrun because the gate did not authorize that phase; no result for them is implied by34passing tests.

### Fixed Stage-A results

**Gate failed: bright-vs-clean AUC < 0.75. Stage B was not run.**

| Views | Clean FPR | Dark AUC | Bright AUC | Dark paired increase | Bright paired increase |
| --- | ---: | ---: | ---: | ---: | ---: |
| all | 13.333% | 0.836389 | 0.598611 | 100.000% | 90.000% |
| full | 8.333% | 0.881944 | 0.604167 | 100.000% | 100.000% |
| quadrants | 14.583% | 0.825955 | 0.600694 | 100.000% | 87.500% |

| Views / condition | Any-activation TPR | Correct-type TPR | Correct type among active |
| --- | ---: | ---: | ---: |
| all / dark | 40.000% | 36.667% | 91.667% |
| all / bright | 18.333% | 16.667% | 90.909% |
| full / dark | 50.000% | 41.667% | 83.333% |
| full / bright | 25.000% | 25.000% | 100.000% |
| quadrants / dark | 37.500% | 35.417% | 94.444% |
| quadrants / bright | 16.667% | 14.583% | 87.500% |

Aggregate correct type among active homogeneous-degraded views: 32/35 = 91.429%. Clean false activations:8/60. The five gate criteria are evaluated on all views; subgroup breakdowns do not replace the aggregate gate.

Paired brightness scores can increase while cross-image discrimination remains weak: bright paired increase is54/60 but AUC is0.598611. This is descriptive evidence of overlapping clean/degraded score distributions, not a causal isolation of content bias. Low thresholded recall is also retained, even though it is not part of the gate.

Only12 held-out source images; crops and synthetic variants are correlated. The18-image metadata-selected subset comes from an existing200-image COCO cache of incompletely documented selection provenance. No claim of representative COCO performance, restoration quality, or spatial-vs-global TTT benefit follows from this audit. All390 scores, clipping values, frozen constants and failed-run receipt remain available.

### Failures, deviations and next action

Two concrete setup/runtime failures preserved. (1) A6000 direct Hugging Face access failed with Errno101/LocalEntryNotFoundError; Windows downloaded the same official weights and server SHA256 matched after transfer. No substitute model. (2) Original run20260912-045406-ttie-t004-a6000 on source1dcda3b exited1 before any held-out score because PyTorch2.4 lacks deterministic upsample_bicubic2d_aa_backward_out_cuda. Minimal source repair disables strict deterministic enforcement only for the two calibration backward checks, measures repeat sensitivity, then restores strict mode for all held-out forwards. Model/crops/prompts/calibration/gate unchanged. Failed receipt retained and frozen calibration is exactly equal between failed/successful runs. No numerical optimizer changes or hidden evaluation reruns.

Read research_log/T004.md for complete predeclarations, selected-pool limitation and interpretation. Successful raw receipt: research_log/remote_runs/20260912-045537-ttie-t004-a6000/; artifacts/audit includes scores.csv/json, config with all prompts/model/manifest, calibration, real_clip_gradient_check.json, summary.json/md and audit_receipt.json. Failed run adjacent20260912-045406-ttie-t004-a6000 retained. Model and image assets remain TTIE/shared/t004, not Git. Recovery logs mirrored under remote project root research_log. No active TTIE job remains.

Recommend research-lead review/acceptance of this negative Stage-A diagnostic and a separately scoped source-trained prompt investigation if desired. The fixed zero-shot bright-exposure signal did not meet the required threshold; no assertion that all frozen VLM signals fail, no restoration or spatial-vs-global semantic benefit established. No additional prompts, percentile/model changes or Stage B/T005 started. Heartbeat stays active15min and awaits actionable feedback/new scoped task; do not rerun T004 just because OPEN persists.

## T005 — DONE (negative fresh relative-signal audit; Stage B not run)

UTC: 2026-09-11T21:58:00Z

PR: https://github.com/word-ky/TTIE/pull/5
Branch: codex/T005-relative-clip.
Protocol/manifest and tested experiment source: b1f7e63e26af574df9a53de1ba288252a9e84968.
Complete evidence: 57d2a97a913b41f297bf1b2dd59150ab90e84ac4.

Received T004 acceptance/PR #4 merge and OPEN T005 in the21:47:28Z heartbeat. Executed the exact fresh relative-signal audit and applied the literal gate. **Stage A fails five of six explicit checks; Stage B is not authorized and was not run.** No learned prompts/T006, response-sign inversion, magnitude/split/prompt/model/crop/threshold tuning or adaptation was attempted after outcomes.

### Implementation, split and frozen constants

Added ttie/relative_clip.py and relative_audit.py; reused existing natural.py, FrozenCLIP, calibration/decision functions and audit_group. No earlier ISP/adaptation/data/model behavior changed. Relative pixel-only API computes current/plus/minus CLIP scores and fixed probe target without references, condition labels or masks; metadata is attached afterward. Exact probes are clamp(image*2**(+.25),0,1) and clamp(image*2**(-.25),0,1). Responses are original dark minus plus dark; original bright minus minus bright. All original/probe score terms are saved, not only response summaries.

Committed fresh research_log/T005_manifest.json before any T005 score. Excluded every18T004 ID, sorted remaining eligible COCO filenames from the same200-image pool, original shorter side>=320, next30 split10calibration/20evaluation. Zero T004 overlap verified. IDs/hashes/dimensions/source URLs/exclusions preserved; no annotation use, no image data committed. The original pool selection provenance remains incompletely known; this is a small fresh diagnostic, not representative COCO sampling.

Same prescribed ViT-B-32/laion2b_s34b_b79k, exact9prompts, fiveviews, sixconditions and preprocessing. Model SHA2561bd3c7172de5b207ceac554f5ab5266166f3b9baccc9af5989bc801016d080ad. No new model/dependency download. Only50clean views from the10fresh calibration IDs determine relative tau=(.011369550973176951,.005873285233974456), scales=(.01,.01), linear95th percentile/population std floor.01. Constants persisted before held-out scoring. Absolute reference on the same fresh inputs uses unchanged T004 thresholds/scales; no retuning.

### Tests, commands and execution

Baseline34tests passed in11.650s. Four new focused tests passed in1.677s (exact probe/response/target formulas, metadata independence, clean-only calibration/fresh split, correct-type TPR/precision gate including no-active undefined precision). Complete local38tests passed in10.246s; remote38tests passed in5.063s. All34T001–T004 regressions retained. Commands: python -m unittest discover -s tests -p test_relative_clip.py -v; python -m unittest discover -s tests -v. git diff --check passed.

A6000 command: bash scripts/run_t005_a6000.sh, invoking python -m ttie.relative_audit --manifest research_log/T005_manifest.json --images /home/wenchang/asdasdsad/wjq/TTIE/shared/t005/images --model-identity /home/wenchang/asdasdsad/wjq/TTIE/shared/t004/model_identity.json --absolute-calibration /home/wenchang/asdasdsad/wjq/TTIE/runs/20260912-045537-ttie-t004-a6000/artifacts/audit/calibration.json --output <artifacts>/audit --device cuda:0.

Release20260912-055303-ttie-t005; run20260912-055307-ttie-t005-a6000, exit0 at21:54:29Z. Actual frozen-CLIP CUDA float32 audit, strict deterministic forward setting, same remote Python3.12.12/torch2.4.0+cu121/torchvision0.19.0/open_clip_torch2.26.1.650unique finite rows:50calibration +600held-out (20images x6conditions x5views). Stored original/probe terms reconstruct responses with maximum error0. Held-out scored once; no evaluation-fed code changes.

Stage-B-only mask/identity/probe-only correction/adaptation-reference replacement/reset tests remain unimplemented/unrun because the gate denied that phase. Existing adaptation regressions and the new pixel-only response/gate/target checks pass; no claim of38tests covering an unimplemented adaptation pilot.

### Complete signal/gate results

**Stage A fails; Stage B was not run.**

| Gate criterion (all views) | Value | Requirement | Pass |
| --- | ---: | ---: | --- |
| Clean relative FPR | 7/100 = 7% | <=15% | Yes |
| Dark response AUC | 0.580600 | >=0.75 | No |
| Bright response AUC | 0.294500 | >=0.75 | No |
| Dark correct-type TPR | 10/100 = 10% | >=30% | No |
| Bright correct-type TPR | 5/100 = 5% | >=30% | No |
| Combined active type precision | 15/21 = 71.429% | >=80% | No |

| Signal / views | Clean FPR | Dark AUC | Bright AUC | Paired dark increase | Paired bright increase |
| --- | ---: | ---: | ---: | ---: | ---: |
| relative / all | 7.000% | 0.580600 | 0.294500 | 56.000% | 29.000% |
| relative / full | 5.000% | 0.520000 | 0.400000 | 40.000% | 45.000% |
| relative / quadrants | 7.500% | 0.597344 | 0.270625 | 60.000% | 25.000% |
| absolute / all | 8.000% | 0.833900 | 0.604000 | 100.000% | 85.000% |
| absolute / full | 0.000% | 0.837500 | 0.632500 | 100.000% | 90.000% |
| absolute / quadrants | 10.000% | 0.831719 | 0.600313 | 100.000% | 83.750% |

| Signal / views / condition | Any-activation TPR | Correct-type TPR | Correct type among active |
| --- | ---: | ---: | ---: |
| relative / all / dark | 10.000% | 10.000% | 100.000% |
| relative / all / bright | 11.000% | 5.000% | 45.455% |
| relative / full / dark | 0.000% | 0.000% | undefined (no active views) |
| relative / full / bright | 10.000% | 10.000% | 100.000% |
| relative / quadrants / dark | 12.500% | 12.500% | 100.000% |
| relative / quadrants / bright | 11.250% | 3.750% | 33.333% |
| absolute / all / dark | 29.000% | 28.000% | 96.552% |
| absolute / all / bright | 9.000% | 7.000% | 77.778% |
| absolute / full / dark | 20.000% | 15.000% | 75.000% |
| absolute / full / bright | 15.000% | 15.000% | 100.000% |
| absolute / quadrants / dark | 31.250% | 31.250% | 100.000% |
| absolute / quadrants / bright | 7.500% | 5.000% | 66.667% |

Absolute reference uses the unchanged T004 model/prompts/thresholds on the SAME fresh T005 images. The relative signal is worse in both AUCs; comparing only with previous T004 images would confound the split. Paired absolute score sensitivity (100% dark,85% bright) does not transfer to the correction-response ranking (56% dark,29% bright). A finite local response is not automatically an exposure-severity score. This audit does not isolate the contribution of saturation/clipping or prove a universal failure of within-image signals.

All650 score rows (50calibration/600heldout), original/plus/minus score terms, fixed targets and clipping remain in scores.csv/json. No response-sign inversion, new probe magnitude, threshold/prompt/model tuning or adaptation was attempted after seeing the result. Fresh30-image manifest excludes every T004 ID; the existing200-image source-pool provenance and correlated20-image held-out views limit population claims.

### Failures, artifacts and next action

One launch transport failure: SSH closed after creating meta.json/run.sh but before tmux launch; local PowerShell reported Cannot call a method on a null-valued expression. Status inspection found no matching TTIE session/process/train.log. Started the same generated run.sh in its intended session/log path using existing workflow helpers, retaining the same run ID. No scientific run had begun, no duplicate evaluation, no code/configuration change. Other jobs remained untouched. Exact failure/recovery receipt in research_log/T005_launch_failure.txt. No runtime or numerical experiment failure after recovery.

Full raw receipt at research_log/remote_runs/20260912-055307-ttie-t005-a6000/: meta/run/train, environment/tests and artifacts/audit/{config.json,calibration.json,calibration_scores.json,scores.csv,scores.json,summary.json,summary.md,audit_receipt.json}. Manifest in T005_manifest.json; full predeclaration/interpretation/receipt in T005.md. Images under remote TTIE/shared/t005/images; same checkpoint under shared/t004. No weights/images in Git. Recovery notes mirrored to remote project-root research_log. No active TTIE job remains.

Recommend research-lead review of PR #5 and this negative fixed relative-signal diagnostic. Same-split absolute/relative comparison does not support the specified correction-response hypothesis. A separately scoped source-trained degradation signal may be considered by the lead, but no learned prompts/T006 started. This does not show that all within-image/VLM signals fail or establish restoration/spatial-TTT/detector utility. Heartbeat remains active15min, awaiting review/new scoped work; do not rerun T005 solely because OPEN persists.

## T006 — DONE (strong learned discrimination/localization; overall gate fails clean FPR)

UTC: 2026-09-11T23:19:21Z

PR: https://github.com/word-ky/TTIE/pull/6
Branch: codex/T006-learned-exposure-prototypes.
Pre-training protocol/manifest and tested source:5788335287232189335ee945c90ddafbeee6269e.
Complete evidence:8c85cb6d301cfbc7ad9474566b580c218820de9f.

Received T005 acceptance/PR #5 merge and OPEN T006 in the23:02:29Z heartbeat. Completed the fixed source-training and fresh homogeneous/mixed audit once. **The literal conjunction fails only clean all-view FPR:21/100=21% exceeds15%.** All other fixed requirements pass. No threshold/prototype/temperature/optimizer/step/split tuning followed evaluation, and no ISP adaptation/T007/CoOp was run regardless of outcome.

### Data, training and leakage boundaries

Fresh T006_manifest.json excludes all48T004/T005 IDs, metadata-sorts remaining eligible original-min-side>=320 files from the existing200-image COCO cache, takes100 of148eligible unused images, and assigns60source_train/20source_calibration/20evaluation. Exact requested sizes, no split deviation. No annotations or image selection by scores. IDs/dimensions/SHA256/URLs/exclusions committed before any training. Original pool selection provenance remains incompletely known;20held-out source images with correlated views do not support representative COCO/population claims. No raw images/checkpoint committed.

Same frozen OpenCLIP ViT-B-32/laion2b_s34b_b79k, exact9prompts and preprocessing/views/conditions. Model SHA2561bd3c7172de5b207ceac554f5ab5266166f3b9baccc9af5989bc801016d080ad. Added image_embeddings seam to existing FrozenCLIP without changing forward arithmetic; new learned_prototypes.py/prototype_audit.py implement only three normalized semantic vectors, source-only training and offline audit. Existing ISP/adaptation untouched. Learned inference accepts only image pixels, frozen encoder/prototypes; decisions use score tensors/calibration before attaching condition/ID metadata. Mixed region labels are attached after all score/decision rows are saved, with exact before/after field equality verified.

Training fixed before outcomes:900frozen512-dimensional source features,300perclass (clean/dark/bright),1536trainable scalars initialized by exact clone of the existing text ensemble means. Full-batch equal-class CE, temperature.07, AdamW lr.005/weight_decay.0001, exactly500updates, seed7. Source records verified to contain exactly the60source_train IDs. CLIP parameters remain frozen/no gradients. Source CE .995049775 -> .353821874;501finite loss samples/500gradient norms retained, no early stopping. Initial/final prototype cosine matrices and alignment are in training_history.json. Initial-to-final class alignment .03523/.15585/.03535 indicates substantial movement from text initialization; no claim that original text semantics are preserved.

Learned artifact prototypes.pt contains initial/raw/final-normalized tensors,20201bytes, SHA256b4b32dbd96c65dcf606ee38d7450ebf348f5731823503b9c71ba15ec78217ac7, verified locally. Model/train config, source cache/history and learned weights/hash saved before calibration/evaluation. Runtime pre_evaluation_checks confirms both CLIP and learned vectors frozen with no accumulated gradients before first held-out score.

Only100clean views from20source_calibration images set the learned95th-percentile thresholds and population-std scales. Learned tau=(.027419920079410076,.001673370413482167), scales=(.07507099353490992,.057845398696933635). For a fair same-split zero-shot reference, its thresholds/scales use the SAME fresh clean IDs/rule:tau=(.03263515159487722,.03038129732012748), scales=(.0311578780744262,.02040387354547658). This baseline calibration choice was explicitly predeclared before training; prompts/prototypes/model unchanged. It differs from T005's retained historic thresholds, so cross-task FPR comparison must respect that distinction. Both constants persisted before any held-out score.

### Tests and actual A6000 run

Baseline38tests passed in13.588s. Prototype/source-isolation5tests passed in5.148s; affected existing CLIP5tests passed in2.412s; mixed audit3tests passed in.025s. Final46tests passed locally in7.561s and remotely in5.102s, including all38T001–T005 regressions. Tests cover exact initialization, prototype-only gradients/frozen CLIP, source-only training/clean-source calibration, deterministic disjoint manifest, pixel-only score/decision invariance, post-hoc localization metadata, correct denominator/region labels and literal mixed gate. git diff --check passed.

Commands: python -m unittest discover -s tests -v (focused patterns test_learned_prototypes.py/test_clip_signal.py/test_prototype_audit.py); bash scripts/run_t006_a6000.sh; python -m ttie.prototype_audit --manifest research_log/T006_manifest.json --images /home/wenchang/asdasdsad/wjq/TTIE/shared/t006/images --model-identity /home/wenchang/asdasdsad/wjq/TTIE/shared/t004/model_identity.json --output <artifacts>/audit --device cuda:0.

Release20260912-071122-ttie-t006; run20260912-071126-ttie-t006-a6000, exit0 at2026-09-11T23:11:54Z. Actual CUDA float32 source feature extraction/prototype training/audit, unchanged Python3.12.12/torch2.4.0+cu121/torchvision0.19.0/open_clip_torch2.26.1 environment.700unique finite score rows:100calibration+600held-out (20images x6conditions x5views).160mixed quadrant views (80pertrue exposure type). No failed numerical run, retraining or evaluation rerun.

### Fixed audit outcome

**Overall gate fails only clean-view false activation:21/100=21% >15%. No ISP adaptation is run.**

| Criterion | Learned result | Requirement | Pass |
| --- | ---: | ---: | --- |
| Clean all-view FPR | 21% | <=15% | No |
| Dark / bright AUC | .9870 / .9366 | each>=.80 | Yes |
| Homogeneous correct-type TPR, dark / bright | 93% /92% | each>=40% | Yes |
| Homogeneous active-type precision | 100% | >=85% | Yes |
| Mixed correct activation recall, dark / bright | 91.25% /95% | each>=40% | Yes |
| Mixed wrong-type activation, dark / bright | 0% /0% | each<=15% | Yes |

| Readout / views | Clean FPR | Dark AUC | Bright AUC | Dark paired increase | Bright paired increase |
| --- | ---: | ---: | ---: | ---: | ---: |
| learned / all | 21.000% | 0.987000 | 0.936600 | 100.000% | 100.000% |
| learned / full | 25.000% | 0.990000 | 0.922500 | 100.000% | 100.000% |
| learned / quadrants | 20.000% | 0.986875 | 0.941094 | 100.000% | 100.000% |
| zero_shot / all | 4.000% | 0.875500 | 0.613800 | 100.000% | 81.000% |
| zero_shot / full | 5.000% | 0.895000 | 0.607500 | 100.000% | 85.000% |
| zero_shot / quadrants | 3.750% | 0.871719 | 0.618125 | 100.000% | 80.000% |

| Readout / views / exposure | Any TPR | Correct-type TPR | Correct type among active |
| --- | ---: | ---: | ---: |
| learned / all / dark | 93.000% | 93.000% | 100.000% |
| learned / all / bright | 92.000% | 92.000% | 100.000% |
| learned / full / dark | 90.000% | 90.000% | 100.000% |
| learned / full / bright | 85.000% | 85.000% | 100.000% |
| learned / quadrants / dark | 93.750% | 93.750% | 100.000% |
| learned / quadrants / bright | 93.750% | 93.750% | 100.000% |
| zero_shot / all / dark | 11.000% | 8.000% | 72.727% |
| zero_shot / all / bright | 10.000% | 10.000% | 100.000% |
| zero_shot / full / dark | 5.000% | 0.000% | 0.000% |
| zero_shot / full / bright | 20.000% | 20.000% | 100.000% |
| zero_shot / quadrants / dark | 12.500% | 10.000% | 80.000% |
| zero_shot / quadrants / bright | 7.500% | 7.500% | 100.000% |

| Mixed readout / condition / true type | Views | Correct recall | Wrong-type rate | Active precision | Margin mean | Margin min /p25 /median /p75 /max |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| learned / all / dark | 80 | 91.250% | 0.000% | 100.000% | 0.211277 | 0.026545 / 0.160090 / 0.206634 / 0.265354 / 0.390412 |
| learned / all / bright | 80 | 95.000% | 0.000% | 100.000% | 0.203356 | -0.021549 / 0.114105 / 0.211364 / 0.269729 / 0.417014 |
| learned / left_right / dark | 40 | 92.500% | 0.000% | 100.000% | 0.210874 | 0.026545 / 0.163605 / 0.206634 / 0.262453 / 0.390412 |
| learned / left_right / bright | 40 | 95.000% | 0.000% | 100.000% | 0.204312 | -0.021549 / 0.129462 / 0.208623 / 0.269729 / 0.417014 |
| learned / quadrants / dark | 40 | 90.000% | 0.000% | 100.000% | 0.211680 | 0.046644 / 0.160090 / 0.216475 / 0.274575 / 0.376905 |
| learned / quadrants / bright | 40 | 95.000% | 0.000% | 100.000% | 0.202399 | -0.021549 / 0.110025 / 0.211447 / 0.267970 / 0.386778 |
| zero_shot / all / dark | 80 | 15.000% | 2.500% | 85.714% | 0.019103 | -0.038418 / -0.000054 / 0.016814 / 0.033747 / 0.070757 |
| zero_shot / all / bright | 80 | 7.500% | 0.000% | 100.000% | 0.029052 | -0.033507 / 0.012587 / 0.028680 / 0.043323 / 0.095283 |
| zero_shot / left_right / dark | 40 | 12.500% | 2.500% | 83.333% | 0.018542 | -0.038418 / -0.001377 / 0.016160 / 0.032606 / 0.070757 |
| zero_shot / left_right / bright | 40 | 10.000% | 0.000% | 100.000% | 0.029547 | -0.021109 / 0.011939 / 0.028680 / 0.046638 / 0.095283 |
| zero_shot / quadrants / dark | 40 | 17.500% | 2.500% | 87.500% | 0.019664 | -0.038418 / 0.003544 / 0.016814 / 0.038451 / 0.070757 |
| zero_shot / quadrants / bright | 40 | 5.000% | 0.000% | 100.000% | 0.028556 | -0.033507 / 0.012587 / 0.028996 / 0.039768 / 0.095283 |

Clean-quadrant false activation: learned20%, zero-shot3.75%. Full-view mixed scores are excluded from localization. Both readouts use the same fresh clean-calibration IDs and95th-percentile/population-std procedure fixed before training; zero-shot text prototypes/prompts are unchanged.

The learned signal substantially improves discrimination and correct localization at the frozen thresholds, while failing the declared clean-content activation ceiling. High AUC or zero mixed wrong-type activation does not establish identity safety. No threshold sweep, early stopping, optimizer/temperature/prototype/split tuning or adaptation followed this result.

Training:900source features (300perclass),1536prototype scalars,500full-batch AdamW updates; source CE .995049775 -> .353821874.501loss samples and500gradient norms, initial/final cosines, learned tensors/hash, calibration and all700score rows are retained. Views/synthetic conditions are correlated;20held-out images from an incompletely documented200-image pool do not establish natural-image population or restoration performance.

### Failures, artifact locations and next action

No scientific/runtime failure. Delivery-only issue: default SFTP-based SCP stalled for several minutes at0bytes on source_features.pt while SSH remained responsive and remote run was finished. Stopped only the verified TTIE receipt-transfer process; the existing workflow's legacy SCP retry completed successfully. Learned weight hash,900x512source cache,700rows and localization invariance checked after transfer. T006_transfer_recovery.txt preserves the event; no experiment was restarted.

Full receipt: research_log/remote_runs/20260912-071126-ttie-t006-a6000/. Includes meta/run/train/environment/tests; artifacts/audit contains config,source_features.pt,training_history.json,prototypes.pt/prototype_identity.json,calibration.json,calibration_scores.json,pre_evaluation_checks.json,scores_before_localization.json,scores.csv/json,summary.json/md and audit_receipt.json. T006.md contains exact predeclaration/provenance/interpretation; T006_manifest.json and T006_tests_cpu.txt support recovery. No raw image dataset or checkpoint committed; derived source features and learned prototype tensor are experiment artifacts. Assets remain TTIE/shared/t006/images and shared/t004/model. Recovery notes mirrored under remote project root research_log. No active TTIE job remains.

Recommend research-lead review of PR #6 and the mixed result: source-trained readout substantially improves ranking/localization, but the fixed clean-content activation criterion still fails and must not be waived. No identity-safe restoration or qualified TTT objective is established. Await the next explicitly scoped research decision; no T007, ISP adaptation, prompt-token learning or threshold repair started. Heartbeat remains active15min and will not rerun completed T006 merely because the inbox remains OPEN.

---

## 2026-09-12T00:19:20Z — T007 DONE: fixed joint rule passes; recall cost retained

**Status: DONE.** PR https://github.com/word-ky/TTIE/pull/7 awaits research-lead review. Branch codex/T007-joint-clean-abstention. Predeclared/tested source **1ad7bacc905840c3eecbf9f551faedb3e0391cf9**; evidence **57c7eeea4fceceb09c20fde659bc5c80460d1cc9**. Received T006 acceptance/merge and T007 from main3ea3d35. No T008 or ISP adaptation started.

### Implementation, frozen inputs and commands

Added ttie/joint_gate.py and joint_audit.py, two focused test files, metadata-only scripts/prepare_t007.py, scripts/run_t007_a6000.sh, README command, protocol/manifest/threshold/test/run receipts. No changes to T001–T006 code or tests. Reuse FrozenCLIP, Prototypes(raw), learned_scores(), raw-winner decisions(), image synthesis/views and existing homogeneous/localization metrics.

Exact prototype SHA256 **b4b32dbd96c65dcf606ee38d7450ebf348f5731823503b9c71ba15ec78217ac7** and original learned tau/scale unchanged. Calibration uses only original20T006 clean source-calibration IDs and exactly5views each: winner rawargmax; e=(d_winner-tau_winner)/scale_winner; per-image max; linear95th percentile across20images. **q_joint=1.053775168916056**. Only activation changes: e>q_joint. float32 scores/baseline decisions unchanged; normalization/quantile arithmetic float64, predeclared before scoring. No sweeps, new source data, retraining or evaluation-driven selection.

Manifest excludes all148T004–T006IDs, ascending numeric eligible original min-side>=320.48unused eligible images found, take first40 exactly as requested, no size deviation.8remain. Original200-image source-pool selection provenance incomplete; no annotations accessed. Metadata-only manifest, complete threshold receipt and tests committed/pushed before any T007 outcome scoring.

Commands: local Python D:/anaconda3/python.exe -m unittest discover -s tests -v and focused test_joint_gate.py/test_joint_audit.py. Existing AutoDL scripts with project .autodl/config.json: deploy -Tag ttie-t007 -Source project; run -Name ttie-t007-a6000 -Cmd 'bash scripts/run_t007_a6000.sh'; explicit-run logs and Copy-FromAutodl. Release20260912-081545-ttie-t007; successful run20260912-081629-ttie-t007-a6000, exit0 at2026-09-12T00:16:50Z. Same A6000/Python3.12.12/torch2.4.0+cu121/OpenCLIP2.26.1 environment, no dependencies added.

### Tests and genuine held-out results

Baseline46tests pass10.556s. New4joint-rule/hash/manifest tests pass.064s; new2persistence/denominator tests pass.033s. Full **52local tests pass8.884s;52remote tests pass5.310s**, including all46prior regressions. Actual CUDA recomputation of original calibration image scores is **bitwise identical, maxdifference0**. CLIP/prototypes frozen and gradient-free.1200unique finite rows (40x6x5); one score vector/type feeds both gates, protected rows persisted before all condition/ID/view/region metadata. Offline receipt verifies exact score/decision preservation, gate/summary recomputation, and exact q reproduction from original calibration. All6AUC comparisons (all/full/quadrants x dark/bright) equal across gates.

**All predeclared T007 criteria PASS.**

| All-view metric | Frozen T006 independent gate | Primary joint gate |
|---|---:|---:|
| Clean FPR |43/200=21.5%|12/200=6%|
| Clean image-any |18/40=45%|8/40=20%|
| Dark/bright AUC |.983650/.930125|.983650/.930125|
| Dark/bright any-activation TPR |93.5%/87%|63.5%/74.5%|
| Dark/bright correct-type TPR |91.5%/86%|62.5%/74.5%|
| Combined active homogeneous type precision |355/361=98.33795%|274/276=99.27536%|
| Mixed dark/bright correct recall |90.625%/87.5%|61.875%/75.625%|
| Mixed dark/bright wrong activation |1.25%/1.25%|1.25%/0%|

Mixed denominators160views per true type, combined LR/quadrants quadrant views only. Full-view cleanFPR27.5%->7.5%, cleanquadrant20%->5.625%; full correctTPRdark/bright90%/87.5%->67.5%/77.5%, quadrant91.875%/85.625%->61.25%/73.75%. Joint mixedLRcorrect61.25%/73.75%, mixedquadrants62.5%/77.5%; wrong1.25%/0% in each. Full all/full/quadrant and mixed condition-specific AUC/anyTPR/precision/margins in summary.json/md.

Tradeoff counts: **31clean view activations removed,0added;58dark/23bright homogeneous correct activations lost;46dark/19bright mixed correct activations lost.**10fewer clean images activate. Passing the conjunction comes with a substantial dark-recall loss. It does not establish restoration quality, downstream utility, universal identity safety or a formal finite-sample coverage guarantee.40heldoutimages and correlated synthetic views from a partially documented pool limit generalization.

### Observed failure, artifacts and recommendation

First run20260912-081549-ttie-t007-a6000 exited1 during tests before ANY fresh scoring: default deployment excludes *.pt, so the T006 prototype was absent. Copied immutable T006 audit artifacts from remote T006 run into the same release using the existing SSH helper; source/weights/config unchanged. Second run above passed and is the only fresh outcome audit. Both run receipts preserved. Future deployment must supply the T006 .pt assets after default exclusions. No other runtime/scientific failure or outcome-driven rerun.

Artifacts: research_log/T007.md, T007_manifest.json, T007_joint_calibration.json, T007_baseline_tests.txt/T007_tests_cpu.txt, and research_log/remote_runs/20260912-081629-ttie-t007-a6000/. The run contains environment/tests/log/meta/run commands; artifacts/audit contains config, pre_evaluation_identity, scores_before_metadata, labeled scores.csv/json, side-by-side summary.json/md and audit_receipt.json. Failed receipt under corresponding081549run. Raw images/checkpoint remain only remote shared assets; no new raw dataset/model committed. Recovery notes mirrored in remote TTIE/research_log. No active TTIE job remains.

Recommend review of PR7 and a separately scoped first learned-signal global-vs-spatial EV+Gamma pilot with direct/discrete-action controls if accepted. Do not extrapolate that pilot's outcome from this audit. Only8eligible unused images remain in current cache; a later >=20fresh-image audit needs a new authorized image pool. No unrequested scope added. Heartbeat remains ACTIVE15min; do not repeat completed T007 while inbox still OPEN.

---

## 2026-09-12T01:03:10Z — T008 PARTIAL: fixed pilot running

T007 acceptance received. T008 implementation/predeclaration source992f7ac8c15a7cb350f027283269f99cfdecf9f0 pushed on codex/T008-semantic-spatial-ttt before outcomes. Complete official image-only COCO val2017 archive downloaded/extracted5000images; deterministic first40eligible numericIDs excluding all188priorIDs, manifestcommitted. No annotations. Frozen T006/T007 assets, originalquadrantmask, two-sidedhinge,2/8parameter EV+gamma state, fixeddirectactions/discretepass and Adam.03max40updates implemented.62tests pass local8.985s/remote5.693s.

A6000 release20260912-090227-ttie-t008; active run20260912-090231-ttie-t008-a6000, bash scripts/run_t008_a6000.sh. Expected240inputs/1680methodrows. Everyinput's7outputs/decisions persisted before clean-reference evaluation. RepresentativeID139fixed across6conditions. No interimoutcome tuning. Fullprecisionoutputs remainremote withperfilehashes; finalmetrics/trajectories/figures/reportfollow afterfullrun.

Observed nvidia-smi NVMLdriver/library mismatch; actualCUDA tensor operations/model run work. No driver/environmentchanges. Permit previouslyobserved unsupported strictantialiasedbicubic backward, seed7/TF32off, actualcalibration-onlygradientpreflight beforefreshscoring. OriginalT006assets suppliedafterdeploy so *.pt exclusion issue avoided. Continue this exactjob, do notlaunchduplicate. No detector/T009.

---

## 2026-09-12T01:17:10Z — T008 DONE: restoration pilot fails four qualification criteria

**Status: DONE.** PR https://github.com/word-ky/TTIE/pull/8 awaits review. Branch `codex/T008-semantic-spatial-ttt`. Predeclared/tested source **992f7ac8c15a7cb350f027283269f99cfdecf9f0**; experiment evidence **89bdcbf7305d600aa4981e81e7494335325fdcfd**; reporting precision **facf43efb79b44506b906d55d0f0be67c8f70ad2**. Source/manifest/protocol/tests were pushed before fresh outcomes. No detector/T009 or post-outcome tuning.

### Implementation and protocol compliance

Added `semantic_ttt.py`, `restoration_metrics.py`, `restoration_pilot.py`, two focused test files, preparation/run scripts, protocol/manifest, and complete metadata/trajectory/figure receipts. No T001–T007 implementation or tests changed. EVGamma reuses the existing bounded ISP mapping/render path with only 2 global or 8 spatial raw EV/gamma coordinates; WB/contrast remain non-trainable identity. Every episode resets raw parameters and Adam. The original four-quadrant scores/mask/winner are computed once and frozen across all methods. The two-sided hinge remains differentiable through frozen CLIP/prototypes to the ISP. Same fixed direct actions, candidate sets/order/ties, and Adam .03/max40/stop1e-8 as requested.

All seven methods ran for all 40 images and all six conditions. Each input's seven full-precision output/raw/grid tensors and gate/trajectory diagnostics were persisted before condition/reference evaluation. No adaptation/control API accepts clean reference, condition, gain, mask, label or evaluation metric. The exact accepted prototype SHA256 `b4b32dbd96c65dcf606ee38d7450ebf348f5731823503b9c71ba15ec78217ac7`, checkpoint, tau/scale and q_joint `1.053775168916056` are unchanged. Scores/ISP float32; normalized loss float64, predeclared.

Complete official COCO val2017 image-only archive obtained because no full directory was available: 815585330 bytes, 5000 JPEGs. No annotations. Numeric filename selection excludes all 188 T004–T007 IDs, requires original min-side >=320, takes first40. Archive/image SHA256, dimensions, URLs and inspected prefix in T008_manifest.json, committed before outcomes. Representative ID139 fixed for every condition. Dataset provenance is now the full official image directory; this small ascending-ID synthetic pilot still does not establish broad generalization.

### Commands, tests and actual run

Local: `D:/anaconda3/python.exe -m unittest discover -s tests -v`, with focused `test_semantic_ttt.py` and `test_restoration_metrics.py` runs. Prior52 baseline tests pass14.899s; new7 invariants pass3.103s; new3 evaluation/manifest tests pass3.082s. **All62 local tests pass8.985s; all62 A6000 tests pass5.693s**, retaining every prior test.

Existing AutoDL deploy/run/log/copy helpers used with project `.autodl/config.json`. Release **20260912-090227-ttie-t008**; run **20260912-090231-ttie-t008-a6000**; command `bash scripts/run_t008_a6000.sh`; exit0 at **2026-09-12T01:11:47Z**. Original T006 audit assets copied into the release after default .pt exclusions before any tests, avoiding the known T007 deployment failure. Data-only download run085514 also exit0. No scientific/runtime failure or experiment rerun.

Actual original source-calibration scores are bitwise identical. Frozen-encoder gradients reach both EV/gamma: raw-gradient norm12.0686569; absolute coordinate sums21.7788200/9.9694595. CLIP/prototypes remain frozen with no .grad or updates. As predeclared from T004's observed limitation, strict deterministic algorithms are disabled for unsupported antialiased-bicubic CUDA backward, seed7/TF32off; repeat gradient max difference **2.52723694e-5**. Do not claim bitwise adaptation repeats. nvidia-smi reports NVML driver/library mismatch, but actual CUDA/model execution works; no drivers/dependencies changed.

Verified **240 input episodes, 1680 unique method rows, 5040 per-input pairwise MSE comparisons, 8565 Adam updates**. All outputs/raw fields/losses/gradients finite and physical fields in bounds. All discrete selections match stored candidate losses and tie rules. All **66 no-active episodes return exact identity** for every method, zero updates. Final frozen assets verified. Local summary/paired recomputation matches saved results exactly; no model rerun.

### Fixed qualification: FAIL

| Criterion | Actual | Verdict |
|---|---:|---|
| Clean spatialTTT mean drift <=.001 |.000816900702|PASS|
| Clean spatialTTT p95 drift <=.005 |.009289116040|FAIL|
| Homogeneous dark MSE reduction >=10% vs identity |61.25897%|PASS|
| Homogeneous bright MSE reduction >=10% vs identity |58.70200%|PASS|
| Pooled heterogeneous MSE reduction >=10% vs identity |16.17631%|PASS|
| Pooled heterogeneous MSE reduction >=15% vs globalTTT |4.45405%|FAIL|
| Pooled heterogeneous MSE >=5% lower than spatial direct |14.73464% HIGHER|FAIL|
| Pooled heterogeneous MSE no more than5% above spatial discrete |14.41182% lower|PASS|
| Dark-region MSE not >10% worse than identity |30.27282% lower|PASS|
| Bright-region MSE not >10% worse than identity |13.65144% HIGHER|FAIL|
| Leakage/immutability |verified|PASS|

Pooled left_right + quadrants, 80 inputs:

| Method | Mean MSE | Mean PSNR dB | Mean per-image recovery |
|---|---:|---:|---:|
|identity|.05992154|12.50577|0|
|global_direct|.06044955|12.45303|-.016079|
|spatial2_direct|.04377792|13.88546|.255237|
|global_discrete|.06367791|12.32680|-.140779|
|spatial2_discrete|.05868620|13.10449|-.044951|
|global_ttt|.05256994|13.08908|.069117|
|spatial2_ttt|.05022844|13.64422|.112563|

Gate reductions use ratios of mean MSE; mean per-image recovery is a different statistic and is reported separately. All method/condition means, medians, p95s, PSNR, recovery, region MSE, loss/step statistics and pairwise comparisons are in summary.json/md and paired.json/csv. Perfect reconstruction PSNR is +infinity, stored null with a separate perfect count rather than included as finite PSNR.

Spatial TTT beats spatial direct on30/80 heterogeneous inputs, loses49, ties1. Direct-minus-TTT MSE mean difference is -.006450521294, median -.003865085542. Left-right spatialTTT MSE .04284468 improves over global .05308801, but quadrants spatial .05761221 worsens versus global .05205186. Fixed spatial direct wins both heterogeneous condition means. Spatial discrete is worse than TTT pooled; do not misstate the failed TTT-vs-direct result as a discrete-search win.

### Failure interpretation and artifacts

There is objective utility on homogeneous exposure, but the fixed spatial TTT mechanism fails qualification. Clean gating abstains on37/40 images; the three active clean images (8/160 active quadrants) cause a tail with max drift .012000620365. Heterogeneous gating activates at least one quadrant on79/80 inputs, so the failure is not simply missing all coverage. Spatial TTT reduces heterogeneous semantic loss **6.070292621 -> .660988056**, below spatial direct's1.806699109, yet produces worse reference MSE. Lower frozen-feature loss is not sufficient restoration evidence.

The dark-region MSE improves .081414989335 -> .056768374646, while bright-region MSE worsens .038444179064 -> .043692361377 (+13.65%). This is an observed harmful region tradeoff; it does not uniquely identify whether objective geometry or coupled spatial actions cause it. Smooth-gradient, report-only: spatialTTT MSE .02690429 versus identity .02042468, worse; it does not alter the primary gate.

Full outputs: **240 SHA256-verified tensor packs, 2,883,922,320 bytes**, retained under remote `TTIE/runs/20260912-090231-ttie-t008-a6000/artifacts/audit/episodes/`. Exact paths/bytes/hashes in output_manifest.json; output_verification.json records the audit. All per-episode decisions/trajectories, metrics, paired rows, summaries, config, preflight/final checks, six fixed representative RGB/EV/gamma panels and environment/test/log/run receipts are committed under local `research_log/remote_runs/20260912-090231-ttie-t008-a6000/`. Only the large outputs.pt files are kept remotely rather than duplicated in Git. Metadata bundle fetched through existing SSH/tar/Copy-FromAutodl; archive cached inside project .autodl. No project artifacts depend solely on session memory. Latest T008/REMOTE/HANDOFF notes mirrored under remote project research_log; no active TTIE job remains.

Recommend research-lead review of PR8 and diagnosis of identity-tail damage, insufficient pooled spatial gain, fixed-action competitiveness and bright-region harm before detector coupling. This controlled pilot does not support the required next-stage claim; no threshold/step/field/weight tuning on these40 images. T008 is complete, not a request to bypass its failed gate. Await a new explicitly scoped task. Heartbeat remains15min and will not rerun completed T008 while OPEN persists.

---

## 2026-09-12T02:36:10Z — T009 PARTIAL: fixed development diagnosis running

T008 accepted negative diagnostic; T009 source/protocol/manifest/tests committed as f911eef036d4de60e5fb9ed410fcec5c132ebc0a on codex/T009-geometry-audit before outcomes. Forty new development_t009 IDs exclude all228 prior IDs, from the complete official image directory; no annotations. Fixed first representative ID4395. Baseline62tests pass16.663s; new8tests and finalfull70tests pass14.950s locally;70tests pass6.039s on A6000.

Active run20260912-103501-ttie-t009-a6000; release20260912-103455-ttie-t009; command bash scripts/run_t009_a6000.sh. Expected200inputs,1200coordinate/state trajectories,80fixed55-pointsurfaces,320renderercomparisonrows. Every semantic trajectory/rawstate is finalized/persisted before separate offline reference-MSE/gradient/oracle calls. ExactT006/T007assets and T008default updatepath reused; trace-on/off parity tests passed. EV/gamma coordinate restrictions and piecewise renderer are explicit labeled controls.

Predeclared pooling/ambiguity resolutions in T009.md: A/B primary active nonclean spatial2 ev_gamma; C primary all5conditions spatial2, withglobal/nonclean secondary reports; E fixedquadrantsrule. Undefined cosine countednonpositive; mediandefinedonly. B reports final-vsidentity and lastupdate-vspenultimate separately, earlierstrictlybetter excludes equalplateaus. D envelope crossing uses fixedgamma1expected-directionEVray, nearwithin.25EV, and both2Dargmins remainreported. Primarygamma rule usesratioofmeanMSE andmeanabsolutesemanticlossreduction. No outcomes used to choose these definitions.

Continue exactjob; no duplicate/retraining/tuning/T010. Data metadata transfer initially needed the existing legacySCP retry; scientificrun has no failure so far. Earlier cosine test used exact floating equality and was repaired to14decimalplaces before outcomes. Same recorded NVMLwarning/actualCUDAworks and nondeterministicbackward allowance as T008. Finaldiagnosis/plots/receipts/PR will follow completion.
