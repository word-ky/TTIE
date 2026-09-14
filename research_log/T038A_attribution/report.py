"""Generate a report and scientific plots from the frozen diagnostic evidence."""
from pathlib import Path
import json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[2];out=root/'research_log/T038A_result'
f=json.loads((out/'stage_a/freeze.json').read_bytes());p=json.loads((out/'stage_a/parity.json').read_bytes());r=json.loads((out/'stage_b/receipt.json').read_bytes());s=json.loads((out/'stage_b/summary.json').read_bytes());v=json.loads((out/'independent_replay.json').read_bytes());a=json.loads((root/'research_log/T038A_archives.json').read_bytes())
g=s['aggregates']['loss_selected'];gap=g['legacy']['positive_dot_fraction']-g['gain']['positive_dot_fraction'];steps=[0,10,20,30,40]
fig,axes=plt.subplots(2,2,figsize=(11,7.5),constrained_layout=True)
for group,color in [('legacy','#3268a8'),('gain','#cc6633'),('total','#438c59')]:
    for col,key in enumerate(['cosine_median','positive_dot_fraction']):
        axes[0,col].plot(steps,[s['aggregates']['step_'+str(step)][group][key] for step in steps],marker='o',color=color,label=group)
        axes[1,col].plot([0,1,2],[s['aggregates'][name][group][key] for name in ['all_selected','loss_selected','nonloss_selected']],marker='o',color=color,label=group)
for col,label in enumerate(['Median cosine','Positive-dot fraction']):
    axes[0,col].set_title('All100 at fixed frozen steps');axes[0,col].set_xlabel('Frozen step')
    axes[1,col].set_title('Accepted selected states');axes[1,col].set_xticks([0,1,2],['All100','PSNR loss29','Non-loss71'])
    for ax in axes[:,col]:ax.set_ylabel(label);ax.grid(alpha=.2);ax.legend()
    axes[0,col].axhline(0,color='grey',lw=.7);axes[1,col].axhline(0,color='grey',lw=.7)
fig.suptitle('T038-A — reference gradient diagnostic only; no optimization or selection changes')
fig.savefig(out/'coordinate_alignment.png',dpi=160);fig.savefig(out/'coordinate_alignment.pdf');plt.close(fig)
text=f'''# T038-A — DONE: {s['classification']}

**REFERENCE_GRADIENT_DIAGNOSTIC_ONLY.** At the exact29 T036 PSNR-loss selected states, gain-group median cosine is **{g['gain']['cosine_median']:.12f}**, positive-dot fraction **{g['gain']['positive_dot_fraction']:.12f} (5/29)**, versus legacy EV+gamma **{g['legacy']['positive_dot_fraction']:.12f} (16/29)**. The difference is **{100*gap:.9f} percentage points**. All three predeclared conditions pass: gain median<=-0.25, gain positive fraction<=35%, and legacy exceeds gain by>=20 percentage points.

The association is specific to the observed loss subset, not evidence that gain is globally harmful: gain has54% positive-dot at all100 selected states and49/71=69.014% in non-loss images. Legacy alignment still collapses late (step30 median-0.304376581, positive22%), so this does not erase the shared legacy-field failure. Of the29 loss cases,13 have valid legacy but invalid gain,11 have both invalid,3 both valid, and2 invalid legacy but valid gain. This is a local raw-coordinate gradient attribution, not a causal intervention, projected-Adam update audit, or deployable repair.

## Frozen inputs and exact bindings

Source **b15a135f48d4f4b84d3a9993790ba49b8f3ab722**, branch `codex/T038A-coordinate-attribution`, PR [#63](https://github.com/word-ky/TTIE/pull/63). Base1adc37d1 after accepted T037 mergec0d84b1d3c7e6af186c28ca736d6ac2bc752d35c. Final evidence SHA is recorded in the main mailbox and delivery receipt.

Exact accepted T036 cohort **{f['cohort_sha256']}**, T036 freeze **{f['prior_freeze_sha256']}**, prior metric CSV **{r['prior_metrics_sha256']}**. Frozen T014 Sobolev checkpoint **{f['assets']['files']['energy']['sha256']}**, including its accepted normalization buffers; CLIP **{f['assets']['files']['clip']['sha256']}**, prototypes **{f['assets']['files']['prototypes']['sha256']}**, gate **{f['assets']['files']['gate']['sha256']}**. All129 staged source/metadata hashes and all checkpoint hashes were verified. Accepted renderer, common-gain energy features, T029 alignment/summarization and model-loading sources were staged as exact Git blobs; no deployable file was modified. Full frozen model tensor hashes are in the Stage-A freeze receipt.

Reuse only the same100 already-reference-used T036/T037 training-development pairs. Stage A does not open normals or read prior metrics/loss IDs. Reference-derived29/71 grouping is read only in Stage B after Stage-A freeze. No new cohort, official test, reference-selected state, retraining, controller or additional action coordinate. The fixed states are0,10,20,30,40 for all100 plus the one accepted selected step36: **501 audited states**. Every image contributes exactly one original selected state.

## Parity, Stage-A freeze and isolated Stage B

Before the first gradient computation, all100 selected common outputs were reconstructed and compared with accepted T036 selected tensors: **100 bit-exact, max error0**, exceeding the required<=1e-6 parity. Preflight completed **{p['completed_utc']}**, with zero gradient calls/reference decodes. Stage A then copied each frozen raw state into the isolated accepted renderer, verified every frozen gate against low-only reconstruction, and evaluated the frozen energy. All501 low-only `g_E`, raw states, output tensors, active masks, gates and energies were saved before reference work. Features and energy values match accepted T036 exactly (max0). Maximum historical backward-vector difference **{f['max_historical_gradient_abs_error']}** is disclosed as auxiliary floating-point backward drift under the accepted nondeterministic CUDA setting; it is not an extra equality gate or an independent scalar-replay error.

Stage A sole A6000 physicalGPU1 run **20260915-064200-ttie-t038a-stage-a**, release20260915-064136-ttie-t038a-attribution, **{f['seconds']:.6f}s**, exit0. It froze at **{f['completed_utc']}**, SHA **{r['stage_a_freeze_sha256']}**. Normal decodes0, optimizer updates0, selection changes0; gradients/states/outputs all finite; raw states remained unchanged during differentiation; all energy/scorer tensors and checkpoint files remained unchanged and their `.grad` fields stayed empty.

Stage B sole A6000 physicalGPU1 run **20260915-064355-ttie-t038a-stage-b**, began **{r['started_utc']}**, first normal open **{min(z['utc'] for z in r['opened_images'] if z['kind']=='normal')}**, completed **{r['completed_utc']}**, **{r['seconds']:.6f}s**, exit0. All100 Stage-A file hashes and source hashes were checked before reference access. It reconstructs the same frozen state with the same renderer and computes `g_R = d mean((output.double()-normal.double())**2) / d raw`; native float32 RGB input/reference pixels, no crop/resize. Stage-B output reconstruction max error **{r['max_stage_a_output_abs_error']}**. No optimizer or checkpoint selection is invoked. All Stage-A files retain their hashes after reference gradients and again during archive creation.

## Group and scalar definitions

The raw tensor is1x3x2x2. Each group includes only cells active in the accepted frozen gate: legacy channels0,1 (EV,gamma), gain channel2 (one post-gamma gain shared over RGB), total allthree channels. Group masks are disjoint and union to total. Per state, total dot equals legacy dot plus gain dot; medians are not additive. The exact accepted T029 routine converts gradients to float64, restricts to active coordinates, and reports dot, Euclidean norms and cosine. A norm<=1e-12 is degenerate (cosine null); positive-dot fraction uses nondegenerate rows as in T029. All reported groups here have zero degenerate/zero-gradient fractions, so fractions use the full stated row count. Valid means strictly `g_E dot g_R > 0`; its negative-energy direction is first-order restorative for MSE. Invalid means nonpositive, not a fitted threshold.

## Per-step and selected-state summaries

Every fixed-step group has100 images. Selected groups are all100, exact29 PSNR-loss, and71 non-loss. Each row reports active-coordinate values; full means/p10/p90 and zero counts are also in `stage_b/summary.json`.

| Set | Group | N | Median cosine | Positive-dot fraction | Median energy norm | Median reference norm | Median dot |
|---|---|---:|---:|---:|---:|---:|---:|
'''
for name in ['step_0','step_10','step_20','step_30','step_40','all_selected','loss_selected','nonloss_selected']:
    for group,z in s['aggregates'][name].items():text+=f"| {name} | {group} | {z['count']} | {z['cosine_median']:.12f} | {z['positive_dot_fraction']:.12f} | {z['energy_norm_median']:.12g} | {z['reference_norm_median']:.12g} | {z['dot_median']:.12g} |\n"
text+='\n![Coordinate alignment](T038A_result/coordinate_alignment.png)\n\n## Sign patterns\n\n| Set | Legacy valid / gain valid | Legacy valid / gain invalid | Legacy invalid / gain valid | Both invalid |\n|---|---:|---:|---:|---:|\n'
keys=['legacy_valid / gain_valid','legacy_valid / gain_invalid','legacy_invalid / gain_valid','legacy_invalid / gain_invalid']
for name,z in s['sign_patterns'].items():text+='| '+name+' | '+' | '.join(str(z.get(k,0)) for k in keys)+' |\n'
text+=f'''
## Independent replay, tests and artifacts

Independent standard-library float64 `math.fsum` dot/norm/cosine replay covers all501 states and1,503 group vectors, including the predeclared30-state subset (indices0,10,...90 at steps0,20,40). Subset max absolute error **{v['subset_max_abs_error']}**, full11334 scalar-check max **{v['all_max_abs_error']}**, both below<=1e-6. Replay also verifies active-channel masks, original selected steps, exact loss IDs, all summaries/sign counts and classification. This is independent scalar replay from saved gradients; gradients were not rerun for this check. Local focused tests **2passed7.92s** and server **2passed1.34s** cover active disjoint groups, dot additivity, negative/degenerate cosine, and all three scientific conditions.

Full Stage-A tensors are retained under the original run and F archive: **{a['archives']['full']['bytes']} bytes**, SHA **{a['archives']['full']['sha256']}**. Compact scalar evidence on both server roots and locally: **{a['archives']['compact']['bytes']} bytes**, SHA **{a['archives']['compact']['sha256']}**; all locally fetched scalars match the archive bytes. `T038A_result/stage_a/` contains parity/freeze receipts, `stage_b/` contains501 raw/gradient/group records, summary and reference-open receipt; run commands/logs and `independent_replay.json` are retained. All working artifacts, recovery notes and the final delivery receipt are mirrored into the project-root `research_log/`.

Commands: Stage A runs `python -m pytest -q tests/test_t038a_attribution.py` then `stage_a.py --accepted <T036 audit> --manifest <T036 manifest> --assets research_log/T036A_assets.json --binding research_log/T038A_source_binding.json --low-root <T036 lows> --out <stage_a>`. Only after successful freeze, `stage_b.py --stage-a <frozen stage_a> --accepted <T036 audit> --manifest <same manifest> --low-root <same lows> --normal-root <same used normals> --out <stage_b>`. Independent local command: `replay.py research_log/T038A_result research_log/T036A_result/audit/metrics.csv`. Exact expanded commands and environment are in the retained run scripts.

No scientific or operational failures/restarts or changed parameter. The existing NVML warning is nonblocking; both stages used CUDA A6000 GPU1. Historical backward drift is disclosed above.

## T038-A-EXEC clarification received during delivery

Research-lead commit d9baaca9de3cc932b7dab06737d8e0635ace6602 was published at22:44:09 UTC while the sole Stage B was already running (it finished22:44:12). It was observed during delivery. This is a clarification of the same T038 audit, not authorization for a duplicate experiment. Its additional numerical replay limits pass on the original frozen receipt: feature0<=1e-6, energy0<=1e-6, historical-gradient2.6226043701171875e-6<=1e-5. No material replay mismatch occurred.

The independent standard-library verifier had already been authored before Stage B and contains no imports/calls of T038/T029 alignment, summarize or classify helpers. The follow-up additionally reconstructs every coordinate mask from a hash-bound export of the frozen Stage-A gate, then replays all501 states and the exact classification independently; maximum error remains4.440892098500626e-16. The gate export reads saved tensors only and opens no image. A single Stage-B entry check now enforces the newly specified replay limits with `BLOCKED_REPLAY_MISMATCH` before reference opening; focused tests cover the actual passing receipt and each failing limit (3passed4.75s locally). This check was added after the completed run, so it is not represented as having executed retroactively. Actual GPU execution remains bound to sourceb15a135f; its original source-binding receipt is preserved. No new gradient, reference evaluation, optimizer run or selection was performed in response to the clarification.

Stop after this attribution audit and await research-lead review. Do not retrain, add a stopping rule, alter gain bounds or consume fresh/test data in this cycle.

{s['classification']}
'''
(root/'research_log/T038A_report.md').write_text(text,encoding='utf-8');print(s['classification'])
