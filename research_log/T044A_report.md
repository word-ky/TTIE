# T044-A: target-free legacy-extrapolation score association audit

**legacy-extrapolation risk association not supported / mixed**. The fixed risk direction fails both predeclared conditions: ROC-AUC **0.249150072850898 < 0.75**, Spearman **+0.483804380438044 > -0.35**. Exactly **29 loss / 71 non-loss** cases reproduce the accepted T036 split. Larger legacy excursion associates with greater paired PSNR improvement here, rather than the hypothesized risk. This is a descriptive association on an already-reference-used cohort, not a causal claim or a deployable rule.

## Scope and physical scalar

Same accepted T036 100 images, frozen Region2 gate, fixed anchor step 10 and each accepted selected common-gain state. No state reselection or gate recomputation. For each active region, d_EV=(EV_selected-EV_step10)/4 and d_gamma=log2(gamma_selected/gamma_step10)/2. D_legacy is sqrt(mean of the squared 2K coordinates), K=active region count. Zero active => 0; actual zero-active images: **0**. Gain is excluded. No image, renderer forward, model, optimizer, threshold or controller is used. The accepted mapping `ttie.isp.physical_parameters` is unchanged: EV=2*tanh(raw_EV), gamma=exp(log(2)*tanh(raw_gamma)); evaluate this same function in float64 from immutable float32 states to make the scalar reproducible. The main score is checked against accepted common-gain physical_grid in a focused test.

## Pre-label bindings and information boundary

Cohort SHA `279c74b335999d6631d436c79ea80e9e2cccfc4b97cb7c0b992829d04cfaf40b`; accepted T036 freeze SHA `46e667ded785e4f3f8ba341d40d00a7e02ca652e2778fcc7ead1750665161be4`. Original trajectory root `/home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-035341-ttie-t036a-common/artifacts/audit`. For every image, scores.json records decision/trajectory file hashes, canonical frozen gate hash, full and legacy-only anchor/selected tensor hashes, selected index, active mask and score. states.pt contains exact frozen anchor/selected snapshots and gates. Source binding pins all imported repository code plus the scalar/statistics/replay implementation; binding SHA `d3c6e10ef45efa74091894105ff56ffa7e7ee154b4040180855797c15357e36d`.

All **100 scores** persisted at **2026-09-15T07:26:11.001945+00:00**, freeze SHA `aa5a7261853c90338a16c0fc85c43190b1cd0dfafa93449b2eea0285800b3195`, score-table SHA `736284a6bf3b95f48a76cf28a431ba11ef668c3ca3ff81b0e81c740ac1aa9a32`, raw-snapshot SHA `a827aa4e82b60638cb2e40238016c110639d43d9165943be003642eb0581f039`. First metric attachment **2026-09-15T07:28:14.382499+00:00**, strictly later. Stage A has no metric/normal/reference-gradient/loss-ID inputs or imports; its file hook logs original-artifact reads and rejects any original audit filename outside freeze/config/decision/trajectory, and rejects image opens. Zero normal, prior metric, prior reference-gradient or loss-ID opens before freeze. Stage B only reads accepted `metrics.csv`, SHA `cdbd7fec76194153db645133d5d35dd43f6f9d852ebbd6674756d77d1ad7aee0`, checks identities/selected steps and exact common-minus-baseline PSNR deltas. Neither stage opens normals. No fresh cohort or official-test access.

## Paired association results

| Group | N | Mean D | Q25 | Median D | Q75 |
|---|---:|---:|---:|---:|---:|
| loss | 29 | 0.120155537021 | 0.087844700268 | 0.120253568093 | 0.143237850243 |
| non_loss | 71 | 0.152978834277 | 0.143333931694 | 0.155491836593 | 0.167906175809 |

![Frozen legacy displacement and paired PSNR](T044A_result/legacy_displacement.png)

AUC treats larger D as greater risk, using average ranks and half-credit for ties. Spearman uses average ranks and their centered correlation; no p-value gate. Quartiles use linear interpolation. No bootstrap, cutoff, inverted score, alternative norm or combined signal was tested.

| Score tail | Image | D_legacy | Paired PSNR delta |
|---|---|---:|---:|
| smallest | Train/Low/low00383.png | 0.034639460847 | +0.764929869010 |
| smallest | Train/Low/low00397.png | 0.037678783188 | -0.153034365495 |
| smallest | Train/Low/low00353.png | 0.068986805254 | -0.302056498331 |
| smallest | Train/Low/low00394.png | 0.073549890725 | -0.466998502132 |
| smallest | Train/Low/low00379.png | 0.074732835186 | -0.844002863904 |
| largest | Train/Low/low00635.png | 0.211257226163 | +0.739105681967 |
| largest | Train/Low/low00238.png | 0.208236357732 | +0.831250381464 |
| largest | Train/Low/low00018.png | 0.205151327316 | +2.546755357882 |
| largest | Train/Low/low00013.png | 0.204022574057 | +4.848330775437 |
| largest | Train/Low/low00580.png | 0.197741141507 | -1.164389457801 |

## Execution and independent replay

Tested source **ba2f1b26e2d09e0eafb3f33b3ed8e078c5d3cd83**, branch `codex/T044A-legacy-drift`, PR https://github.com/word-ky/TTIE/pull/69. Release `20260915-152431-ttie-t044a-audit-fixed`. Stage A run `20260915-152458-ttie-t044a-stage-a`, score generation 0.299032456s. Stage B run `20260915-152656-ttie-t044a-stage-b`, association 0.006045695s, independent replay 0.219842268s. Both actual executions exit0; light CPU state arithmetic, no GPU model work was needed. Exact commands and single-thread environment are retained in both run/run.sh receipts.

Independent replay imports no main T044 score/statistics/verdict helper. It reloads original hash-bound trajectories/decisions and frozen snapshots, checks every gate and state is exact, reconstructs physical EV/gamma with Python math, recomputes all100 scores, AUC via all **2059** positive/negative pairs, Spearman with explicit tie ranks and math.fsum, paired deltas, descriptive summaries and tail identities. **210 scalar checks**, max discrepancy **8.326672684688674e-17 <=1e-10**, final verdict exact, PASS. Independent replay also checks freeze precedes metric attachment.

Baseline renderer test: 1 passed in12.37s. New local tests: 2 passed in10.41s; server tests: 2 passed in2.35s. Tests cover physical mapping, active masks, zero-active case, gain exclusion, tied AUC/Spearman and both verdict boundaries.

## Failures, deviations, artifacts and next step

Preparation: initial sparse-checkout add command used an unsupported option, leaving a test dependency absent; corrected sparse inclusion, then baseline passed. Two local test processes aborted inside NumPy cov/corrcoef called by scipy spearmanr, including a single-thread retry. Replaced covariance with the algebraically equivalent explicit centered-rank sums; local and server tests pass. Initial source b4e322d and release152254 were superseded before any experimental run or metric access. Report generation also had one unterminated string, corrected before rendering. These are implementation/environment repairs, with no scientific definition or threshold change.

SSH connection timed out once during each stage launch, before either process started. Checked absence of tmux/logs before resuming: Stage A reused its exact generated run.sh; Stage B reconstructed the same wrapper from the persisted command in meta.json. Each scientific stage ran exactly once; no partial score or label computation preceded these recoveries. No scientific deviations, optimizer updates, state/selection changes, rerendering, fresh cohort, normal opens or official-test access.

Evidence archive 109482 bytes, SHA `eae5dbb57c9cb0f6974e680833e985af08f290ceb61a663449d5163c817addb8`, verified locally and under both home/F `shared/t044a`. Contains score freeze, exact raw snapshots/gates, paired table, statistics, independent replay and both run logs. Original T036 trajectories remain unchanged. Source and evidence are delivered by PR; a single completion report is appended directly to the main Codex mailbox. Project-local reports/plots/recovery are retained under research_log.

Stop after this association audit. The predeclared large-excursion risk hypothesis is unsupported and the observed association points the other way; do not turn D into a trust radius, invert it, fit a threshold or implement a controller in this cycle. Research lead should reassess why productive legacy movement and local field unreliability coexist before specifying another intervention.

legacy-extrapolation risk association not supported / mixed
