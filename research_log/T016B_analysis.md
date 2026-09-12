# T016-B: frozen Sobolev hard-boundary selection audit

Verdict: **negative**, 0/5 clauses pass.

Development-only reuse of 120 old episodes; nine hard boundaries and fixed saved corner actions. No training, optimization or new images.

Selection finalized and hashed before reference access: `{'finalized_utc': '2026-09-12T17:38:16.358698+00:00', 'selection_sha256': '99dbb10260e5045d5c5deb1d53b5f68d42567a02d5a1b898b234e6b5d113b866', 'config_sha256': 'e4c8d8ddfe16333fa4c766f6e1ecca2388e89a7f2ce9ba36c3537017cd666e64', 'episodes': 120, 'energies': 1080, 'frozen_assets_unchanged': True, 'reference_access': False, 'source_sha': '7352c073a0d34af3aee63b6817fa02b63f79f860'}`

| Group | Selected MSE | Region2 | Hard oracle | Full oracle | T015 oracle | Selected/Region2 | Selected/hard oracle | Selected/full oracle | Selected/T015 oracle | Hard/full oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spatial_pool | 0.0378572608073 | 0.0350435737482 | 0.0325635645189 | 0.0325076995068 | 0.0343645194507 | 1.08029109928 | 1.16256501297 | 1.16456289992 | 1.10163800956 | 1.00171851632 |
| left_right | 0.0359704593895 | 0.0333970155101 | 0.0321709857788 | 0.0321141692344 | 0.0330589110497 | 1.07705610337 | 1.11810249263 | 1.12008064499 | 1.08807151377 | 1.00176920486 |
| quadrants | 0.0383517022361 | 0.0309838496498 | 0.0309821883566 | 0.0309821883566 | 0.0305849858909 | 1.23779655109 | 1.23786292287 | 1.23786292287 | 1.25393885657 | 1 |
| offset_left_right_40 | 0.0392496207962 | 0.0407498560846 | 0.0345375194214 | 0.0344267409295 | 0.0394496614113 | 0.96318428008 | 1.13643427362 | 1.14009109595 | 0.994929218452 | 1.00321780363 |

## Fixed acceptance clauses

- spatial_improves_region2_3pct: False
- spatial_within_hard_oracle_5pct: False
- offset_improves_region2_5pct: False
- left_right_no_more_than_1pct_worse: False
- quadrants_no_more_than_1pct_worse: False

## Counts, ranks, margins and regret

Candidate order: ((0.4, 0.4, 0.0), (0.4, 0.5, 0.0), (0.4, 0.6, 0.0), (0.5, 0.4, 0.0), (0.5, 0.5, 0.0), (0.5, 0.6, 0.0), (0.6, 0.4, 0.0), (0.6, 0.5, 0.0), (0.6, 0.6, 0.0)). Exact energy and oracle ties choose first lexicographic index. Spearman uses average ranks; constant ranks give null. Null denominator/counts stay explicit. Regret is selected MSE minus nine-hard oracle MSE.

### spatial_pool

Selected counts: [19, 15, 6, 23, 25, 14, 4, 10, 4]; oracle counts: [27, 2, 12, 23, 41, 14, 0, 1, 0].

Disagreement: 0.725; outside oracle tie set: 0.725. Energy/oracle minimum-tied episodes: 14/12.

Energy margin: {'count': 120, 'null_count': 0, 'mean': 0.026922967036565146, 'quantiles': {'min': 0.0, 'p05': 0.0, 'p25': 0.0024842023849487305, 'p50': 0.012310385704040527, 'p75': 0.0417250394821167, 'p95': 0.10427869558334343, 'max': 0.16879773139953613}}

Spearman: {'count': 120, 'null_count': 9, 'mean': 0.2397858357298814, 'quantiles': {'min': -0.95, 'p05': -0.6916666666666667, 'p25': -0.13333333333333333, 'p50': 0.36666666666666664, 'p75': 0.675, 'p95': 0.9333333333333333, 'max': 1.0}}

| Selected index | Count | Mean regret | Regret quantiles |
|---|---:|---:|---|
| 0 | 19 | 0.0032093121033 | {'min': 0.0, 'p05': 0.0, 'p25': 0.0, 'p50': 0.0022018998861312866, 'p75': 0.005467217415571213, 'p95': 0.009013593755662427, 'max': 0.014877578243613243} |
| 1 | 15 | 0.00564160092423 | {'min': 0.0011644288897514343, 'p05': 0.00152834951877594, 'p25': 0.004485004581511021, 'p50': 0.005295503884553909, 'p75': 0.007142465561628342, 'p95': 0.009728226065635679, 'max': 0.011699430644512177} |
| 2 | 6 | 0.002465403794 | {'min': 0.0, 'p05': 0.0, 'p25': 6.214715540409088e-06, 'p50': 2.7074944227933884e-05, 'p75': 0.0021047682967036963, 'p95': 0.009655407629907131, 'max': 0.011941678822040558} |
| 3 | 23 | 0.00663803027862 | {'min': 0.0, 'p05': 0.0, 'p25': 0.00024383887648582458, 'p50': 0.006559707224369049, 'p75': 0.009159342385828495, 'p95': 0.01820180797949433, 'max': 0.01842641457915306} |
| 4 | 25 | 0.00208664737642 | {'min': 0.0, 'p05': 0.0, 'p25': 0.0, 'p50': 1.0026618838310242e-05, 'p75': 0.003102727234363556, 'p95': 0.009454353898763642, 'max': 0.018831221386790276} |
| 5 | 14 | 0.00345187640882 | {'min': 0.0, 'p05': 0.0, 'p25': 2.9707560315728188e-05, 'p50': 0.0012946445494890213, 'p75': 0.004286383278667927, 'p95': 0.011306899692863226, 'max': 0.011968912556767464} |
| 6 | 4 | 0.0126813827083 | {'min': 0.004662672057747841, 'p05': 0.005523431766778231, 'p25': 0.00896647060289979, 'p50': 0.010740535333752632, 'p75': 0.014455447439104319, 'p95': 0.0225565199740231, 'max': 0.0245817881077528} |
| 7 | 10 | 0.011364230793 | {'min': 0.0027477890253067017, 'p05': 0.004923493042588234, 'p25': 0.008015834726393223, 'p50': 0.01135227782651782, 'p75': 0.01240225019864738, 'p95': 0.020265134656801805, 'max': 0.026623639278113842} |
| 8 | 4 | 0.0143287996762 | {'min': 0.004315610975027084, 'p05': 0.0059450590983033185, 'p25': 0.012462851591408253, 'p50': 0.01535829808562994, 'p75': 0.01722424617037177, 'p95': 0.02127124248072505, 'max': 0.02228299155831337} |

### left_right

Selected counts: [9, 3, 0, 7, 8, 6, 3, 3, 1]; oracle counts: [2, 0, 0, 23, 2, 13, 0, 0, 0].

Disagreement: 0.675; outside oracle tie set: 0.675. Energy/oracle minimum-tied episodes: 5/5.

Energy margin: {'count': 40, 'null_count': 0, 'mean': 0.018566972017288207, 'quantiles': {'min': 0.0, 'p05': 0.0, 'p25': 0.0010949373245239258, 'p50': 0.006077408790588379, 'p75': 0.024433553218841553, 'p95': 0.0809709906578064, 'max': 0.08308625221252441}}

Spearman: {'count': 40, 'null_count': 3, 'mean': 0.32207207207207206, 'quantiles': {'min': -0.9333333333333333, 'p05': -0.6866666666666666, 'p25': -0.15, 'p50': 0.43333333333333335, 'p75': 0.85, 'p95': 1.0, 'max': 1.0}}

| Selected index | Count | Mean regret | Regret quantiles |
|---|---:|---:|---|
| 0 | 9 | 0.00394837910102 | {'min': 0.0, 'p05': 0.0, 'p25': 0.002677038311958313, 'p50': 0.004407661035656929, 'p75': 0.006083205342292786, 'p95': 0.006950227171182632, 'max': 0.0070798806846141815} |
| 1 | 3 | 0.00614791798095 | {'min': 0.005295503884553909, 'p05': 0.005390501581132412, 'p25': 0.005770492367446423, 'p50': 0.006245480850338936, 'p75': 0.006574125029146671, 'p95': 0.00683704037219286, 'max': 0.006902769207954407} |
| 2 | 0 | null | {} |
| 3 | 7 | 6.96682504245e-05 | {'min': 0.0, 'p05': 0.0, 'p25': 0.0, 'p50': 0.0, 'p75': 0.0, 'p95': 0.0003413744270801541, 'max': 0.00048767775297164917} |
| 4 | 8 | 0.000676333904266 | {'min': 0.0, 'p05': 3.509316593408585e-06, 'p25': 1.597357913851738e-05, 'p50': 9.733811020851135e-05, 'p75': 0.0005339602939784527, 'p95': 0.002806756924837826, 'max': 0.0036545805633068085} |
| 5 | 6 | 9.06954519451e-05 | {'min': 0.0, 'p05': 0.0, 'p25': 0.0, 'p50': 0.0, 'p75': 8.912268094718456e-05, 'p95': 0.00034871441312134266, 'max': 0.00042534247040748596} |
| 6 | 3 | 0.0132151767612 | {'min': 0.004662672057747841, 'p05': 0.0052365118637681006, 'p25': 0.00753187108784914, 'p50': 0.01040107011795044, 'p75': 0.01749142911285162, 'p95': 0.023163716308772564, 'max': 0.0245817881077528} |
| 7 | 3 | 0.0121245762954 | {'min': 0.011428502388298512, 'p05': 0.01153081189841032, 'p25': 0.011940049938857555, 'p50': 0.0124515974894166, 'p75': 0.012472613248974085, 'p95': 0.012489425856620074, 'max': 0.01249362900853157} |
| 8 | 1 | 0.0155379977077 | {'min': 0.015537997707724571, 'p05': 0.015537997707724571, 'p25': 0.015537997707724571, 'p50': 0.015537997707724571, 'p75': 0.015537997707724571, 'p95': 0.015537997707724571, 'max': 0.015537997707724571} |

### quadrants

Selected counts: [3, 7, 1, 8, 11, 3, 0, 5, 2]; oracle counts: [0, 0, 0, 0, 39, 0, 0, 1, 0].

Disagreement: 0.725; outside oracle tie set: 0.725. Energy/oracle minimum-tied episodes: 2/0.

Energy margin: {'count': 40, 'null_count': 0, 'mean': 0.03634818196296692, 'quantiles': {'min': 0.0, 'p05': 0.00034790039062500007, 'p25': 0.0066762566566467285, 'p50': 0.026235103607177734, 'p75': 0.05146944522857666, 'p95': 0.12781881093978878, 'max': 0.14537310600280762}}

Spearman: {'count': 40, 'null_count': 0, 'mean': 0.26415569415042095, 'quantiles': {'min': -0.6166666666666667, 'p05': -0.26583333333333325, 'p25': -0.029166666666666667, 'p50': 0.2833333333333333, 'p75': 0.5916666666666667, 'p95': 0.8508333333333332, 'max': 0.9}}

| Selected index | Count | Mean regret | Regret quantiles |
|---|---:|---:|---|
| 0 | 3 | 0.0084805060178 | {'min': 0.0022018998861312866, 'p05': 0.0028179138898849487, 'p25': 0.005281969904899597, 'p50': 0.008362039923667908, 'p75': 0.011619809083640575, 'p95': 0.014226024411618708, 'max': 0.014877578243613243} |
| 1 | 7 | 0.00729213788041 | {'min': 0.004447814077138901, 'p05': 0.0044701283797621725, 'p25': 0.005612471140921116, 'p50': 0.007382161915302277, 'p75': 0.008145308122038841, 'p95': 0.010854628682136533, 'max': 0.011699430644512177} |
| 2 | 1 | 0.011941678822 | {'min': 0.011941678822040558, 'p05': 0.011941678822040558, 'p25': 0.011941678822040558, 'p50': 0.011941678822040558, 'p75': 0.011941678822040558, 'p95': 0.011941678822040558, 'max': 0.011941678822040558} |
| 3 | 8 | 0.0122755571501 | {'min': 0.0060054101049900055, 'p05': 0.0063059570267796515, 'p25': 0.0073735760524868965, 'p50': 0.012237267568707466, 'p75': 0.016972470795735717, 'p95': 0.01841399734839797, 'max': 0.01842641457915306} |
| 4 | 11 | 0 | {'min': 0.0, 'p05': 0.0, 'p25': 0.0, 'p50': 0.0, 'p75': 0.0, 'p95': 0.0, 'max': 0.0} |
| 5 | 3 | 0.0112173942228 | {'min': 0.01073283888399601, 'p05': 0.010754598118364812, 'p25': 0.010841635055840015, 'p50': 0.010950431227684021, 'p75': 0.011459671892225742, 'p95': 0.01186706442385912, 'max': 0.011968912556767464} |
| 6 | 0 | null | {} |
| 7 | 5 | 0.00957943014801 | {'min': 0.0075826868414878845, 'p05': 0.007594063133001328, 'p25': 0.0076395682990550995, 'p50': 0.009144634008407593, 'p75': 0.01127605326473713, 'p95': 0.012058577314019202, 'max': 0.012254208326339722} |
| 8 | 2 | 0.0132993012667 | {'min': 0.004315610975027084, 'p05': 0.005213980004191399, 'p25': 0.008807456120848656, 'p50': 0.013299301266670227, 'p75': 0.0177911464124918, 'p95': 0.021384622529149053, 'max': 0.02228299155831337} |

### offset_left_right_40

Selected counts: [7, 5, 5, 8, 6, 5, 1, 2, 1]; oracle counts: [25, 2, 12, 0, 0, 1, 0, 0, 0].

Disagreement: 0.775; outside oracle tie set: 0.775. Energy/oracle minimum-tied episodes: 7/7.

Energy margin: {'count': 40, 'null_count': 0, 'mean': 0.02585374712944031, 'quantiles': {'min': 0.0, 'p05': 0.0, 'p25': 0.002050638198852539, 'p50': 0.011310815811157227, 'p75': 0.02140986919403076, 'p95': 0.1319318056106567, 'max': 0.16879773139953613}}

Spearman: {'count': 40, 'null_count': 6, 'mean': 0.12156862745098039, 'quantiles': {'min': -0.95, 'p05': -0.8508333333333333, 'p25': -0.43333333333333335, 'p50': 0.20833333333333331, 'p75': 0.5333333333333333, 'p95': 0.9066666666666665, 'max': 1.0}}

| Selected index | Count | Mean regret | Regret quantiles |
|---|---:|---:|---|
| 0 | 7 | 0 | {'min': 0.0, 'p05': 0.0, 'p25': 0.0, 'p50': 0.0, 'p75': 0.0, 'p95': 0.0, 'max': 0.0} |
| 1 | 5 | 0.00302705895156 | {'min': 0.0011644288897514343, 'p05': 0.0012684062123298645, 'p25': 0.0016843155026435852, 'p50': 0.0022706175222992897, 'p75': 0.004907270893454552, 'p95': 0.005068383738398552, 'max': 0.005108661949634552} |
| 2 | 5 | 0.000570148788393 | {'min': 0.0, 'p05': 0.0, 'p25': 0.0, 'p50': 2.4858862161636353e-05, 'p75': 2.9291026294231415e-05, 'p95': 0.0022431334480643265, 'max': 0.002796594053506851} |
| 3 | 8 | 0.00674782018177 | {'min': 0.0032325834035873413, 'p05': 0.003553725965321064, 'p25': 0.004801323637366295, 'p50': 0.007310841232538223, 'p75': 0.008707042317837477, 'p95': 0.009235675539821386, 'max': 0.009413786232471466} |
| 4 | 6 | 0.00779258552939 | {'min': 0.003102727234363556, 'p05': 0.0033364174887537956, 'p25': 0.004261525347828865, 'p50': 0.005155149847269058, 'p75': 0.00919949822127819, 'p95': 0.016741860192269087, 'max': 0.018831221386790276} |
| 5 | 5 | 0.00282598286867 | {'min': 0.0006004571914672852, 'p05': 0.0008781321346759797, 'p25': 0.0019888319075107574, 'p50': 0.00346258282661438, 'p75': 0.003544297069311142, 'p95': 0.004335855692625045, 'max': 0.004533745348453522} |
| 6 | 1 | 0.0110800005496 | {'min': 0.011080000549554825, 'p05': 0.011080000549554825, 'p25': 0.011080000549554825, 'p50': 0.011080000549554825, 'p75': 0.011080000549554825, 'p95': 0.011080000549554825, 'max': 0.011080000549554825} |
| 7 | 2 | 0.0146857141517 | {'min': 0.0027477890253067017, 'p05': 0.003941581537947059, 'p25': 0.008716751588508487, 'p50': 0.014685714151710272, 'p75': 0.020654676714912057, 'p95': 0.025429846765473482, 'max': 0.026623639278113842} |
| 8 | 1 | 0.0151785984635 | {'min': 0.015178598463535309, 'p05': 0.015178598463535309, 'p25': 0.015178598463535309, 'p50': 0.015178598463535309, 'p75': 0.015178598463535309, 'p95': 0.015178598463535309, 'max': 0.015178598463535309} |

No reference information enters the scorer. Evaluation reads only immutable saved selections and accepted T016-A MSE values; no rendering occurs in evaluation.

Stop for research-lead review. No recalibration, feature extension, boundary predictor, new candidate family or fresh evaluation was started.

## Execution, provenance and numerical limits

Scientific source: `7352c073a0d34af3aee63b6817fa02b63f79f860`, branch `codex/T016B-boundary-selection`, PR17 https://github.com/word-ky/TTIE/pull/17. The branch starts from task main9a960db6; state update6a1aa4f0 confirms unchanged T016-B scope. Only frozen T016-A initializer/renderer were ported from05e7dcf; no T016-A screen or accepted metric was rerun. All scientific scripts and tests remained unchanged after the frozen commit.

One completed scoring run: `20260913-013748-ttie-t016b-assets-ready`, actual release `20260913-013400-ttie-t016b-selection`, exit0 at2026-09-12T17:38:19Z, A6000 physicalGPU1/logicalcuda0. Exact command is in run.sh/meta.json; it calls `bash scripts/run_t016b_a6000.sh 7352c073a0d34af3aee63b6817fa02b63f79f860` from the explicit release, with CUDA_VISIBLE_DEVICES=1. Preparation, scoring and evaluation each invoke the unchanged Git provenance guard before their data/model work; all65 scientific files match actual Git blobs. No gradients, optimizer, action fitting, head training, new image IDs or candidate-family changes.

Preparation uses the fixed120 old directory keys to read only accepted label-free outputs.pt, saved gate decisions and label_free_receipt.json. It extracts identity pixels/corners/gates, never reconstructing pixels from a clean image. The scorer receives this packaged input index and frozen assets; episode directory strings are output join keys only and are not supplied to features or the scoring kernel. No condition/image ID/reference MSE/clean-pixel input is accepted by that kernel. Source manifest and original receipts are read solely by the existing asset provenance checks. Candidate current evidence is recomputed serially for each of nine rendered outputs. Feature/head code and normalization are unchanged.

Selection SHA256 `99dbb10260e5045d5c5deb1d53b5f68d42567a02d5a1b898b234e6b5d113b866`; config SHA256 `e4c8d8ddfe16333fa4c766f6e1ecca2388e89a7f2ce9ba36c3537017cd666e64`. Selection finalized2026-09-12T17:38:16.358698+00:00; separate evaluator opened references2026-09-12T17:38:18.728917+00:00, and verified selection/config hashes. Every attached reference MSE matches the accepted T016-A Git table at ee5d8fdaf3ab48ee7ad3654d45bdc65419be8367, tableSHA `6091a0c928f115940997a647693b6571d8608131e7f9567a75882f0233b9c41e`. No rerender or selection change occurred after reference access.

Frozen Sobolev headSHA `c3d1eef9f20af163e1268823cf16d3fdaed315e7723fc94db3b1138ad1336521`; value-only receipt controlSHA `df6c5af2610e741cf03a59239135a82204550fab3bcbf9e408e553521ce7b69c`; T006 prototypesSHA `b4b32dbd96c65dcf606ee38d7450ebf348f5731823503b9c71ba15ec78217ac7`. Exact model/checkpoint, gate/head/source receipt hashes, normalization, 28-feature schema, all65 code hashes, current CLIP scores and 28 features for every candidate are in scoring/config.json and selection.json.

Verification confirms120 original identity tensors, physical corners and saved gates unchanged;1080 finite energies and lexicographic choices; all per-condition reference joins/means/ratios/counts/clauses;111 nonconstant Spearman values independently matched SciPy. Nine undefined constant-rank correlations remain null. Quantiles and conditioned regret were separately checked locally using NumPy. Source/table audit passes.

**Canonical numeric limitation:** fresh inference-only scoring is not bitwise equal to the saved gradient-enabled T015 trajectory evaluations. Observed maximum absolute differences across120 canonical candidates are CLIP scores `3.650784492492676e-7`, 28-features `6.318092346191406e-6`, energy `4.0531158447265625e-6`. Physical pixels/corners/gates and source/assets are unchanged. The exact cause of this small numerical drift was not independently isolated; different inference/gradient execution paths are a possible explanation, not an experimentally established attribution. An initial extra verifier assertion of exact zero failed; the verifier now reports measured differences explicitly rather than claiming bitwise equivalence. No required task threshold was changed, and no scientific score/selection/code was modified or rerun in response. Fourteen winner margins are zero (and the only margins <= twice that observed canonical energy difference); the canonical observation is not a bound on all candidate numerical errors. Do not claim bitwise reproduction or numerical robustness beyond measured evidence.

Focused local checks:4 baseline energy-core tests PASS4.432s;2 kernel tests PASS.195s; final4 leakage/equivalence/tie/rank/acceptance tests PASS.324s; py_compile PASS. Full repository suite not rerun because the task requests focused tests and no donor scientific module changed. The first baseline command used the wrong unittest module import form and failed to resolve sibling fixtures; ordinary discovery fixed the invocation without source edits. Both logs are retained.

Two failed pre-score launch attempts are retained. Run013458 stopped before input access because Windows CRLF index-info text indexed filenames with a trailing carriage return; only deployment index text was corrected to LF, and `.git/index.failed-crlf` was preserved. Run013638 passed source guard and prepared inputs but stopped at asset receipt verification because existing deployment exclusions omit checkpoint files. The two accepted T014 heads and frozen T006 prototypes were copied from the accepted T015 release; receipt checks then passed. Neither failed launch produced candidate energies or opened reference tables. No deployment workflow redesign or provenance bypass occurred.

Compact run/evidence archiveSHA `b8fad98acb548346574a4012669836c1c8dbf3be09da5aacab103a07aa175340` matches server/local. It contains finalized scores, evaluation, preparation index, post-run verification and all three launch logs/metadata; it excludes duplicated input tensors. The tensor packs remain on A6000 under the completed run; original accepted T015 pixels remain read-only on F. Git metadata transfer packSHA `55f4ebf7187c16da852988c71f25a2d37b4bc0df189d3164079d7bbe7db395f2`,65 scientific files/6081 objects, preserves the original runtime guard.

## Interpretation for the research lead

The nine-hard oracle is only0.17185% above the full27 oracle, so excluding softness retains almost all previously observed headroom. The frozen energy selector nevertheless yields8.03% higher spatial MSE than canonical Region2 and16.26% above the nine-hard oracle. It harms left/right by7.71% and exact quadrants by23.78%; offset improves3.68%, below the required5%. All five fixed clauses fail.

The selector disagrees with the reference oracle on87/120 episodes (72.5%, also outside all minimum-MSE ties). Mean within-episode Spearman is0.2398 over111 nonconstant episodes; mean/median winner energy margin are0.02692/0.01231. These are diagnostics only and were never used to change the selection rule. The evidence does not justify deployment of raw frozen-energy boundary selection. It also does not invalidate T014's previously qualified within-basis energy trajectory or T016-A's reference-only geometric headroom. Stop and let the next research review decide what to test; no predictor, recalibration, feature extension or fresh run has been initiated.
