# T014 Stage A — derivative supervision passes the source-calibration contract

The repaired, frozen experiment passes all eight Stage-A clauses on the20 source-calibration images. Relative to the same-source value-only control, Sobolev supervision improves calibration identity-gradient alignment and reduces heterogeneous restoration MSE by16.58%. The new trajectory's reference-only oracle also clears the discrete/fixed16 margins. This supports the derivative-supervision hypothesis on this predeclared development split; it is not yet a fresh-evaluation qualification.

Frozen source: `f861b2c6ffde6d017cb174ef8e00cb75701bf5e1`. Run: `20260912-181047-ttie-t014-stage-a-repaired`, exit0 at2026-09-12T11:01:44Z. Source manifest SHA256 `4dbf8c604bc44574a5823ec8e22142c8ac39574b6103c4bbd0a69647f15a2257`;80train/20calibration images excluding508prior IDs. All100 are permanent development; the fresh split must exclude608prior/source IDs.

| Stage-A clause | Observed | Required | Result |
|---|---:|---:|---|
| Positive calibration gradient cosine |73/74 =0.9864864864864865|>=0.80|PASS|
| Median calibration gradient cosine |0.9360590709945287|>=0.50|PASS|
| Clean p95 MSE |0.0002394345123320852|<=0.005|PASS|
| Homogeneous dark / identity MSE |0.5591337468311788|<=0.65|PASS|
| Homogeneous bright / identity MSE |0.41086523819069704|<=0.65|PASS|
| Heterogeneous / projected discrete MSE |0.9263470330290793|<=0.95|PASS|
| Heterogeneous / frozen semantic16 MSE |0.9186562220330877|<=0.95|PASS|
| Heterogeneous / same-source value-only MSE |0.8342483020057921|<=0.95|PASS|

## Training fit and calibration transfer are separate

Both heads use identical7346source rows from400episodes, the same28features/SiLU architecture/train-only normalization/seed7/batch order/100epochs/final checkpoint.7248active nonzero-reference-gradient rows contribute directional diagnostics and, only for the Sobolev head, directional supervision.98all-inactive episodes retain one value-only identity row. The value-only training objective is unchanged T013 Huber; the matched control was not selected as a fallback.

| Source-train diagnostic only | Value-only | Sobolev |
|---|---:|---:|
| Standardized value Huber |0.033193279057741165|0.051005665212869644|
| Positive gradient cosine fraction (saved float32 statistic) |0.8652042150497437|0.9976544976234436|
| Median gradient cosine |0.46205344796180725|0.9724292755126953|
| Mean directional loss |0.2913585901260376|0.025412224233150482|

The Sobolev head trades some scalar fit accuracy for better source derivative fit. These training values are not a gate. On74active non-clean calibration episodes, value-only has59positive/15negative cosines, positivefraction0.7972972972972973 and median0.4246446532217347. Sobolev has73positive/1negative, positivefraction0.9864864864864865 and median0.9360590709945287. The paired calibration deltas are +0.18918918918918926 in positive fraction and +0.511414417772794 in median cosine. Full distributions and per-condition counts are in the actual run's `final_distributions.json/.md`; training and calibration sections are explicitly separated.

## Trajectory quality and selector regret

Primary/oracle heterogeneous MSE ratio is1.044737586056045. The oracle/discrete ratio is0.8866791483267122 and oracle/fixed16 is0.8793176720108989. Thus the actual new checkpoints contain states beating both controls by more than5%; the label-free minimum-energy selector itself still passes both margins. This differs from the earlier selection-only limitation, but the earlier tasks used different development images and are not a paired causal comparison. The same-source value-only control is the causal comparison in T014.

All four learned-energy geometries/control paths save complete trajectories. The run contains12400energycheckpoints/12000updates and2557old-semantic checkpoints;25calibration inputs bypass adaptation as all-inactive. Exact selected-step histograms, projected-update fractions and final movable-boundary fractions are reported from saved trajectories, not used to change optimization or checkpoint choice. The five fixed representative panels use the first calibration ID48555, with clean plus all10reported methods; no representative was selected after inspecting outcomes.

## Reproducibility and limits

- All125local tests pass135.151s;125A6000 tests pass37.822s. Original T006 calibration preflight is bitwise equal. Frozen CLIP/prototype/T007 identities verified; both head hashes are unchanged across calibration.
- Remote audit verifies4600file hashes, every stored-pixel MSE, cached feature construction, both GPUhead scores/selections, train-only normalization/final fitting diagnostics, calibration alignment and projection/summary calculations.18020inactive-region checks are exact. Full41,933,631,120image bytes remain in the original server run; complete small records include source Jacobians and reference gradients.
- Source J/g_ref are training-only. Test-time APIs take only the frozen28features/pixels/state flow; replacement-reference/metadata tests preserve trajectories/decisions/outputs. Direct cached-Jacobian/autograd equivalence was checked on a fixed actual-CLIP fixture, not re-run at all7346states; all saved derivative records were checked for integrity/schema/finiteness/feature correspondence/masks and exact final-fit replay.
- Initial run `20260912-180228-ttie-t014-stage-a` failed before fitting/calibration because no-grad CLIP feature caching differed from the grad-enabled derivative forward. The repaired cache uses the same differentiable forward, preserving the original1e-6 check and all scientific settings. Failed evidence is retained; research lead accepted this repair in `0c971a56b72ec8bdaeee7f71bf8a3f433d72252f`.
- The same fixed raw bank states are rendered with the frozen masked Region2 function so feature/value/derivative records describe one function.517source rows differ from inherited bank pixels by at most5.960464477539063e-08. Calibration direct control is unchanged. CUDA backward nondeterminism remains disclosed; no strict repeated-CUDA-trajectory bitwise claim.

## Next authorized action

Commit the immutable receipt and both model files, verify their actual Git blobs, then create/commit the deterministic40-image fresh manifest before scoring. The twelve Stage-B clauses and report-only offset stress remain unchanged. No retuning or new task is authorized.

The existing/home partition has only about36GB free after Stage A, while fresh complete trajectories need more. New Stage-B artifacts will use the writable persistent mount `/media/wenchang/F/wjq/TTIE/runs`, with about19.4TB free. Execution metadata/logs and frozen code remain in the existing project, all prior artifacts are preserved, and only the existing artifact-directory environment setting changes.

## Verified calibration method table

| Method | Clean mean MSE | Dark MSE | Bright MSE | Heterogeneous MSE |
|---|---:|---:|---:|---:|
| identity | 0 | 0.0866946687922 | 0.0453317561187 | 0.0644856445957 |
| region2_direct | 8.98603117093e-05 | 0.0592909015715 | 0.0170994870481 | 0.0382898011245 |
| region2_discrete_projected | 0.000239434512332 | 0.0530567258596 | 0.0198342825519 | 0.0364867357304 |
| region2_ttt_projected | 0.000204961956479 | 0.0544820488431 | 0.0196465004236 | 0.0371277014608 |
| fixed_step_source | 0.000239434512332 | 0.0539458294865 | 0.0193003657274 | 0.0367921955767 |
| region2_ttt_energy_value_only | 0.000239434512332 | 0.055375020043 | 0.0246598666476 | 0.0405147715705 |
| global_ttt_energy_sobolev | 0.00144763151184 | 0.0389302096562 | 0.0132990188315 | 0.0618799178861 |
| bilinear2_ttt_energy_sobolev | 0.000146961864084 | 0.047698653955 | 0.0174361400539 | 0.0439394287765 |
| region2_ttt_energy_sobolev | 0.000239434512332 | 0.0484739149921 | 0.0186252427753 | 0.0337993793888 |
| oracle_best_sobolev_checkpoint | 9.19808628419e-19 | 0.0483272586949 | 0.0156117436185 | 0.0323520277627 |

| Method | Projected update fraction | Final movable-boundary fraction | Selected-step histogram |
|---|---:|---:|---|
| region2_ttt_energy_value_only | 0.972 | 0.800970873786 | {'0': 25, '7': 1, '12': 10, '13': 2, '14': 1, '15': 3, '16': 1, '20': 1, '21': 3, '23': 1, '24': 1, '25': 1, '26': 3, '27': 1, '28': 3, '29': 4, '30': 2, '31': 2, '32': 3, '33': 2, '35': 3, '36': 2, '37': 3, '38': 3, '39': 3, '40': 16} |
| global_ttt_energy_sobolev | 0.532 | 0.533333333333 | {'0': 25, '5': 1, '7': 1, '10': 1, '12': 16, '13': 1, '14': 1, '15': 1, '18': 1, '19': 1, '20': 2, '21': 1, '22': 2, '23': 2, '24': 1, '25': 2, '29': 1, '30': 1, '31': 4, '32': 2, '33': 1, '34': 4, '35': 2, '36': 1, '37': 3, '38': 5, '39': 2, '40': 15} |
| bilinear2_ttt_energy_sobolev | 0.931 | 0.631067961165 | {'0': 25, '12': 18, '13': 4, '14': 1, '20': 1, '23': 1, '24': 2, '26': 1, '28': 2, '29': 1, '30': 4, '31': 4, '32': 2, '33': 1, '34': 1, '35': 1, '36': 4, '37': 1, '38': 1, '39': 5, '40': 20} |
| region2_ttt_energy_sobolev | 0.936 | 0.650485436893 | {'0': 25, '12': 19, '13': 3, '14': 1, '15': 1, '18': 1, '25': 2, '26': 2, '27': 1, '28': 2, '30': 3, '31': 1, '32': 2, '33': 3, '34': 3, '36': 2, '37': 3, '38': 4, '40': 22} |

Local downloaded evidence verifies3100small hashes,1000calibration method rows and148alignment records. Calibration summary recomputation is exact (maxabsolute difference0). Final archive SHA2017ce30df9f851b142cb7e668ab26d30d03665ca11558a3e4cd9eb8404f0156 matches server.

Sobolev checkpoint SHA256c3d1eef9f20af163e1268823cf16d3fdaed315e7723fc94db3b1138ad1336521; value-only checkpoint SHA256df6c5af2610e741cf03a59239135a82204550fab3bcbf9e408e553521ce7b69c. Frozen receipt also verifies against the actual original release code/manifest/schema/Jacobian/loss/normalization/asset identities.
