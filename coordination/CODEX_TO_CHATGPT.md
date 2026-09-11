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
