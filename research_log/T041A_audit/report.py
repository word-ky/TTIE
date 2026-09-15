"""Generate the completed T041 report and figure from saved scalar evidence."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path('research_log');root=r/'T041A_result'
s=json.loads((root/'stage_b/summary.json').read_bytes());f=json.loads((root/'stage_a/freeze.json').read_bytes());b=json.loads((root/'stage_b/receipt.json').read_bytes());p=json.loads((root/'stage_a/parity.json').read_bytes());v=json.loads((root/'independent_replay.json').read_bytes());a=json.loads((r/'T041A_archives.json').read_bytes())
assert v['status']=='PASS' and v['classification']==s['classification']
lines=['| Population | Gain | Group | Count / nondegenerate | Median cosine | Positive dot | Median energy norm | Median reference norm | Median dot | Degenerate fraction |','|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
for gain in ['1.25','1.75']:
 for population,groups in [('source frozen',s['source_baselines'][gain]['groups']),('real selected',s['by_gain'][gain])]:
  for name,d in groups.items():lines.append(f"| {population} | {gain} | {name} | {d['count']} / {d['nondegenerate']} | {d['cosine_median']:.12f} | {d['positive_dot_fraction']:.12f} | {d['energy_norm_median']:.12g} | {d['reference_norm_median']:.12g} | {d['dot_median']:.12g} | {d['either_zero_fraction']:.12f} |")
lines.extend(['','| Gain | Group | Real-minus-source positive dot (pp) | Real-minus-source median cosine |','|---|---|---:|---:|'])
for gain,groups in s['target_minus_source'].items():
 for name,d in groups.items():lines.append(f"| {gain} | {name} | {d['positive_dot_fraction']*100:.9f} | {d['cosine_median']:.12f} |")
table='\n'.join(lines)
fig,axs=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
for ax,key,title in zip(axs,['positive_dot_fraction','cosine_median'],['Positive-dot fraction','Median gradient cosine']):
 for pop,color in [('source','#2171b5'),('real','#d94801')]:
  vals=[(s['source_baselines'][g]['groups'] if pop=='source' else s['by_gain'][g])['total'][key] for g in ['1.25','1.75']]
  ax.plot([1.25,1.75],vals,marker='o',color=color,label='Frozen source (7,346 states)' if pop=='source' else 'Real selected (100 states)')
 ax.set(title=title,xlabel='Fixed common gain',xticks=[1.25,1.75]);ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Total active-coordinate alignment at matched fixed gain')
fig.savefig(root/'matched_gain_alignment.png',dpi=170);fig.savefig(root/'matched_gain_alignment.pdf');plt.close(fig)
assets='\n'.join(f"- {k}: `{d['sha256']}`" for k,d in f['assets']['files'].items())
base='\n'.join(f"- Gain {g}: `{d['path']}`, SHA `{d['sha256']}`" for g,d in s['source_baselines'].items())
text=f"""# T041-A — DONE: {s['classification']}

**REFERENCE_GRADIENT_DIAGNOSTIC_ONLY.** At fixed gain1.75, all100 real selected states have total-group positive-dot **0.37** versus frozen source **0.796633554084**, a **42.663355408 percentage-point deficit**; real median cosine **-0.185847992683** versus source **0.763459378857**, a **0.949307371541 deficit**. Both exceed the predeclared20pp/0.25 thresholds. The single primary verdict is **{s['classification']}**. The exact task-decimal constants define the gate; matched-source descriptive deltas use the full-precision accepted summaries. No thresholds were fitted.

This supports a real target/selected-state distribution effect beyond common gain value alone. It does not isolate image-domain content shift from selected legacy EV/gamma/feature-state extrapolation. The populations also differ:100 real selected states versus7346 canonical source states (7248 nondegenerate); source states are correlated within80 source images, so this is a prescribed descriptive diagnostic, not an independent-sample significance claim.

## Reuse and provenance

Executed/tested source **63e0cb7770c5f2f7c4d85244cf7a83a5a065e87a**, branch `codex/T041A-fixed-gain-real-state`, PR [#66](https://github.com/word-ky/TTIE/pull/66), baseef4969ef after accepted T040 mergeeabe742e333f97a0706117ce92abfc3b238964d1. Evidence/main-mailbox SHAs are retained in final delivery. Reuse T038 low-only original-selected-output preflight and FixedObjective gate verification, T036 model loading, T039 common-gain raw conversion, and T029/T038 active-coordinate alignment/aggregate conventions. No deployable module is changed.

Exact T036100 cohort SHA **{f['cohort_sha256']}**, accepted T036 freeze SHA **{f['prior_freeze_sha256']}**. Source accepted audit is `/home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-035341-ttie-t036a-common/artifacts/audit`. Its config hash and assets are checked against the accepted freeze; each common decision/output/trajectory/field file is checked against the accepted per-image hashes. Each original selected step must match both frozen metadata and the earliest target-free minimum-energy score, without making any new choice. Source bindings are retained for every deployed code/config file used by Stage A; exact Git bytes were deployed.

{assets}

Only the selected legacy EV/gamma raw channels and exact gate are reused. Each active cell's gain is replaced with the spatially uniform prescribed1.25 or1.75 through exp(log2*tanh(raw)); inactive gain stays1. No image selection, optimizer, interpolation, sweep, new cohort or source-state resampling. Exactly100x2=200 new probes. The original selected output is reconstructed for parity only and receives no new reference/gradient audit. Every gate score/active/winner/evidence tensor matches accepted T036 exactly.

Accepted source comparisons are hash-bound, never recomputed:
{base}

Stage A binds the comparison metadata but does not open those source summary files; Stage B checks both full source-summary hashes/values after the complete low-only freeze. Historical real PSNR/SSIM, metrics.csv and loss-ID files are not opened in either stage. The optional29/71 grouping is intentionally omitted; all100 states determine the sole verdict.

## Freeze and reference boundary

All100 original selected outputs are **bit-exact**, max absolute error **{max(x['selected_output_max_abs'] for x in p['rows'])}**, before the first probe-gradient call. Stage A run **20260915-122826-ttie-t041a-stage-a**, release20260915-122752-ttie-t041a-fixed-gain, used A6000 physicalGPU1 and took **{f['seconds']:.6f}s**, exit0. It saved all200 native outputs, raw states, features, energies, g_E, active masks/gates and output hashes. Entire evidence froze **{f['completed_utc']}**, SHA **{b['stage_a_freeze_sha256']}**. Before freeze: normal decodes0, prior real metric/loss-ID file opens0, optimizer updates0, selection changes0. Frozen scorer/energy parameters and normalization remain unchanged, all tensors/scalars finite, raw states unchanged during gradient calls.

Only after freeze, Stage B run **20260915-123014-ttie-t041a-stage-b** started **{b['started_utc']}**; first normal **{min(x['utc'] for x in b['opened_images'] if x['kind']=='normal')}**; completed **{b['completed_utc']}**, duration **{b['seconds']:.6f}s**, exit0 on GPU1. It accesses only the same100 T036 normals already reference-used by T037/T038 under shared/t036a/normal and their lows. Exact accepted native float32 RGB decoding is reused, with float64 RGB-MSE accumulation for isolated reference gradients. **All200 output hashes and tensors match Stage A bit-exactly**; all100 Stage-A tensor hashes remain unchanged afterward and at archive. No reference quantity changes any state/output. No official-test access, fresh cohort, retraining, controller or deployable decision change.

## Matched-gain alignment

Use active raw-coordinate legacy EV+gamma, common gain and disjoint total masks. Float64 dot/norm/cosine, norm<=1e-12 degeneracy, null undefined cosine and nondegenerate denominators exactly follow T029/T038. Total-dot additivity is independently verified. All200 real probe rows are nondegenerate in every group. Positive dot indicates locally restorative negative-energy direction for RGB-MSE, not a finite-update performance guarantee.

{table}

![Matched fixed-gain total alignment](T041A_result/matched_gain_alignment.png)

At gain1.25, real total positive-dot53%/median cosine0.076331 are already much worse than source96.6887%/0.916274. At1.75, real legacy is only37% positive with median cosine-0.220698, whereas real gain remains71% positive with median0.309086. Thus the all100 fixed-gain total deficit is not an all100 gain-specific failure; the legacy field is substantially misaligned in these selected states. This does not contradict T038's different question about the original29-loss subset and original spatial gain states.

Full means/p10/p90, norm/dot medians and zero/degenerate fractions are retained in summary.json; every raw state/gradient/mask/group scalar and output hash is retained in states.json. Full features/outputs and g_E remain in Stage-A tensor archives. Independent stdlib replay imports no main alignment/summarize/classify path, reconstructs masks from separately frozen gates, checks all200 identities/selected steps, recomputes600 group vectors, all aggregates, source-baseline hashes and deltas, and the single primary verdict. **{v['scalar_checks']} scalar checks**, max error **{v['max_abs_error']} <=1e-6** (includes comparison of full-precision baseline values to task-rounded constants). Baseline3tests passed9.75s; T0412local tests7.08s and2server tests1.38s passed. No further scientific modification after reference access.

## Failures, artifacts and next step

No GPU stage failures or restarts; one Stage A and one Stage B. Initial local test command was issued before the new sparse worktree was checked out, so it found no files and ran no tests; checkout fixed that mechanical preparation issue. No disk-full issue in this task. The existing NVML warning is nonblocking. No scientific deviations or extra probes. Optional29/71 analysis is not implemented, as permitted.

Full F archive **{a['archives']['full']['bytes']} bytes**, SHA **{a['archives']['full']['sha256']}**; compact evidence **{a['archives']['compact']['bytes']} bytes**, SHA **{a['archives']['compact']['sha256']}**, both server roots and locally. All exact commands, environment/run logs, original-parity/freeze/reference receipts, vectors and plots are retained; Git-exact source/evidence/mailbox recovery is mirrored locally and in home/F shared/t041a. Stage-A command is in stage_a_run/run.sh; Stage-B command in stage_b_run/run.sh; independent check: `python research_log/T041A_audit/replay.py research_log/T041A_result`.

Stop after this audit. Recommend research-lead review of the real selected legacy/feature-state distribution before choosing any repair; do not infer a pure content-domain cause or automatically retrain/design a controller from this diagnostic. No new experiment is launched in this cycle.

{s['classification']}
"""
(r/'T041A_report.md').write_text(text,encoding='utf-8');print(s['classification'])
