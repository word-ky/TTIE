"""Generate T042 report and figures from independently verified frozen evidence."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path('research_log');root=r/'T042A_result'
s=json.loads((root/'stage_b/summary.json').read_bytes());f=json.loads((root/'stage_a/freeze.json').read_bytes());b=json.loads((root/'stage_b/receipt.json').read_bytes());v=json.loads((root/'independent_replay.json').read_bytes());a=json.loads((r/'T042A_archives.json').read_bytes())
assert v['status']=='PASS' and v['classification']==s['classification']
sets=[('T041 selected',s['baselines']['selected']['groups']),('T042 step10',s['groups']),('T040 source',s['baselines']['source']['groups'])]
lines=['| State population | Group | Count / nondegenerate | Median cosine | Positive dot | Median energy norm | Median reference norm | Median dot | Degenerate fraction |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
for population,groups in sets:
 for name,d in groups.items():lines.append(f"| {population} | {name} | {d['count']} / {d['nondegenerate']} | {d['cosine_median']:.12f} | {d['positive_dot_fraction']:.12f} | {d['energy_norm_median']:.12g} | {d['reference_norm_median']:.12g} | {d['dot_median']:.12g} | {d['either_zero_fraction']:.12f} |")
lines.extend(['','| Comparison | Group | Positive-dot change (pp) | Median-cosine change |','|---|---|---:|---:|'])
for name,groups in s['step10_minus_baseline'].items():
 for group,d in groups.items():lines.append(f"| step10 minus {name} | {group} | {100*d['positive_dot_fraction']:.9f} | {d['cosine_median']:.12f} |")
table='\n'.join(lines)
fig,axs=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
for ax,key,title in zip(axs,['positive_dot_fraction','cosine_median'],['Positive-dot fraction','Median gradient cosine']):
 vals=[g['total'][key] for _,g in sets];ax.bar(['Selected','Step 10','Source'],vals,color=['#d94801','#238b45','#2171b5']);ax.set(title=title);ax.axhline(0,color='black',linewidth=.6);ax.grid(axis='y',alpha=.2)
 for i,val in enumerate(vals):ax.annotate(f'{val:.3f}',(i,val),xytext=(0,5 if val>=0 else -14),textcoords='offset points',ha='center',fontsize=9)
 ax.margins(y=.15)
fig.suptitle('Total active-coordinate alignment; common gain fixed at 1.75')
fig.savefig(root/'legacy_state_alignment.png',dpi=170);fig.savefig(root/'legacy_state_alignment.pdf');plt.close(fig)
baselines='\n'.join(f"- {name}: `{d['path']}`, SHA `{d['sha256']}`" for name,d in s['baselines'].items())
assets='\n'.join(f"- {name}: `{d['sha256']}`" for name,d in f['assets']['files'].items())
text=f"""# T042-A — DONE: {s['classification']}

**REFERENCE_GRADIENT_DIAGNOSTIC_ONLY.** On the same100 T036 real images/gates at fixed common gain1.75, substituting each trajectory's step10 legacy EV+gamma raises total positive-dot from **37% to94% (+57pp)** and median cosine from **-0.185847992683 to0.685276543900 (+0.871124536583)**. Both exceed the fixed +20pp/+0.25 criterion. The single verdict is **{s['classification']}**. No threshold, state or baseline was fitted after observing references.

This within-image substitution supports late real legacy/feature-state extrapolation as a major contributor and weakens a pure content-only account of T041. It does not establish a deployable checkpoint/stopping rule: these are isolated gradient probes with gain fixed1.75, not original step10 outputs, new optimization trajectories or a PSNR/SSIM comparison. No permission to deploy step10 is inferred.

## Provenance and fixed substitution

Executed/tested source **951e06671a74c3c2b91bd92646182029c723b7d0**, branch `codex/T042A-step10-legacy`, PR [#67](https://github.com/word-ky/TTIE/pull/67), base80078e23 after accepted T041 merge3bb7aa0a8561a03df573fde5f27bdbf04b420d8a. Evidence/main-mailbox SHAs are in final delivery. Reuse accepted T041 low-only gradient/evaluation loops, T036 model loading, T038 gate/alignment grouping, T039 probe_raw and T029 scalar conventions. No deployable source module changes.

Exact T036 cohort SHA **{f['cohort_sha256']}**, accepted original freeze **{f['prior_freeze_sha256']}**, original audit `/home/wenchang/asdasdsad/wjq/TTIE/runs/20260915-035341-ttie-t036a-common/artifacts/audit`. Config/assets and all100 common trajectory/decision/output/field file hashes match the original freeze. Before the first learned-gradient calculation, provenance.json binds all100 low names, original per-file hashes, fixed legacy step10 and exact float32 legacy-slice hashes. Each later probe rechecks its step10 slice hash and copies those eight raw EV/gamma coordinates unchanged. Gate scores/active/winner/evidence match accepted T036 tensors exactly.

{assets}

Use exactly **100 new probes, step10 legacy + gain1.75**. The original step10 common-gain channels are ignored. Active common gain is set uniformly1.75 through accepted exp(log2*tanh(raw)); inactive gain stays1. No selected-state recomputation, step0/20/30/40 gradient probes, step/gain sweep, interpolation, optimization, new selection or new cohort. The same100 pairs were already reference-used in T037/T038/T041.

Baselines are bound rather than recomputed:
{baselines}

T041 selected total positive-dot0.37 and median cosine-0.18584799268346364 are checked exactly. The primary test uses the task's equivalent explicit thresholds total>=0.57 and median>=0.06415200731653636. Source1.75 and the legacy/gain groups are descriptive only. Optional per-image sign-transition and29/71 loss grouping are not implemented; no loss-ID/PSNR/SSIM file is opened.

## Isolation and execution

Stage A run **20260915-133013-ttie-t042a-stage-a**, release20260915-132936-ttie-t042a-step10, executed on A6000 physicalGPU1 in **{f['seconds']:.6f}s**, exit0. All100 raw states, native outputs, features, energies, masks/gates, g_E and output hashes were frozen **{f['completed_utc']}**, SHA **{b['stage_a_freeze_sha256']}**. Before freeze: normal decodes0, prior real metric/loss-ID/reference-gradient file opens0, optimizer updates0, selection changes0. Stage A reads baseline path/hash metadata only; actual reference-derived baseline summary files are excluded from its source-file read checks and opened by Stage B only. All tensors/scalars finite; model parameters/normalization unchanged; raw states unchanged during gradients.

Stage B run **20260915-133153-ttie-t042a-stage-b** started **{b['started_utc']}**. First reference-related baseline open **{min(x['utc'] for x in b['baseline_opened'])}**, first normal **{min(x['utc'] for x in b['opened_images'] if x['kind']=='normal')}**, both after complete Stage-A freeze. Completed **{b['completed_utc']}**, duration **{b['seconds']:.6f}s**, exit0 on GPU1. Only the same100 authorized T036 normals under shared/t036a/normal and their lows were decoded. Exact native float32 RGB input handling and isolated float64 mean RGB squared-error gradient are reused.

**All100 Stage-B output hashes and tensors are bit-exact** to Stage A; all100 Stage-A tensor hashes remain unchanged after evaluation/archive. Step10 legacy hashes also remain exact. No reference-dependent state/gain/gate/output/model/decision changes, no optimizer updates/selection changes, no fresh cohort or official-test access.

## Alignment and comparisons

Float64 active-coordinate dot/norm/cosine with norm<=1e-12 degeneracy exactly follows T029/T038. Legacy EV+gamma, gain and their disjoint total are reported separately. Positive-dot fraction and cosine exclude degenerate rows; all100 T042 rows are nondegenerate in each group. Positive dot denotes a locally restorative negative learned-energy gradient for RGB-MSE, not a finite optimizer-step quality guarantee.

{table}

![Legacy-state substitution at fixed gain](T042A_result/legacy_state_alignment.png)

The largest change is in legacy alignment: positive-dot37% to94%, median cosine-0.220698 to0.686961. Gain changes71% to78% and median0.309086 to0.535115. Relative to source1.75, step10 real total positive-dot is14.336645pp higher but median cosine0.078183 lower; this descriptive cross-population comparison adds no gate and is not a claim that real outperforms source on image quality. Source states are correlated canonical bank states; real probes are one fixed state per image.

Independent stdlib replay imports no main alignment/summarize/classify path. It rebuilds active masks from hash-frozen gates, checks all100 identities and step10 provenance, reproduces legacy float32 hashes using struct.pack, recomputes300 gradient group vectors, dot additivity, all aggregates/baseline deltas, verifies the accepted baseline file hashes, and independently recomputes the primary verdict. **{v['scalar_checks']} scalar checks**, max error **{v['max_abs_error']} <=1e-6**, PASS. Full mean/p10/p90, norm/dot medians and zero/degenerate fractions are in summary.json; all per-state raw/g_E/g_R/masks/scalars/output hashes are in states.json; full output/feature tensors remain in Stage A.

Baseline T0412tests passed19.65s; T0422local tests passed16.50s; T0422server tests passed1.35s. No scientific changes after reference access. One Stage A and one Stage B, no failures/restarts/deviations. Existing NVML warning is nonblocking. Narrow local sparse checkout completed without disk problems.

## Artifacts and next step

Full F archive **{a['archives']['full']['bytes']} bytes**, SHA **{a['archives']['full']['sha256']}**; compact **{a['archives']['compact']['bytes']} bytes**, SHA **{a['archives']['compact']['sha256']}**, verified on both server roots and locally. Exact source commands/environment/run logs, provenance/freeze/reference receipts, scalar vectors and plots are retained. Git-exact source/evidence/mailbox recovery including baseline JSON bytes is mirrored in project research_log and home/F shared/t042a. Commands are in stage_a_run/run.sh and stage_b_run/run.sh; independent check is `python research_log/T042A_audit/replay.py research_log/T042A_result`.

Stop after this diagnostic and await research-lead review. The next method choice must address state/feature reliability without assuming a universal deployable step10 policy. No controller, retraining or new experiment is implemented in this cycle.

{s['classification']}
"""
(r/'T042A_report.md').write_text(text,encoding='utf-8');print(s['classification'])
