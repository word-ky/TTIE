"""Render a report from completed, independently replayed T040 receipts."""
from pathlib import Path
import json,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path('research_log');root=r/'T040A_result'
s=json.loads((root/'stage_b/summary.json').read_bytes());f=json.loads((root/'stage_a/freeze.json').read_bytes());b=json.loads((root/'stage_b/receipt.json').read_bytes());v=json.loads((root/'independent_replay.json').read_bytes());a=json.loads((r/'T040A_archives.json').read_bytes());base=s['frozen_T039_gain125_baseline']
assert v['status']=='PASS' and v['classification']==s['classification']
sets=[('T039 frozen 1.25',base['groups'])]+[(g,s['by_gain'][g]) for g in ['1.5','1.75']]
lines=['| Gain | Group | Count / nondegenerate | Median cosine | Positive dot | Median energy norm | Median reference norm | Median dot | Degenerate fraction |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
for gain,groups in sets:
 for name,d in groups.items():
  lines.append(f"| {gain} | {name} | {d['count']} / {d['nondegenerate']} | {d['cosine_median']:.12f} | {d['positive_dot_fraction']:.12f} | {d['energy_norm_median']:.12g} | {d['reference_norm_median']:.12g} | {d['dot_median']:.12g} | {d['either_zero_fraction']:.12f} |")
lines.extend(['','| Gain | Legacy-minus-gain positive-dot pp | Legacy-minus-gain median cosine | Gain positive-dot change vs frozen1.25 pp | Gain cosine change vs frozen1.25 |','|---|---:|---:|---:|---:|'])
for gain in ['1.5','1.75']:
 l,g=s['by_gain'][gain]['legacy'],s['by_gain'][gain]['gain'];bg=base['groups']['gain']
 lines.append(f"| {gain} | {100*(l['positive_dot_fraction']-g['positive_dot_fraction']):.9f} | {l['cosine_median']-g['cosine_median']:.12f} | {100*(g['positive_dot_fraction']-bg['positive_dot_fraction']):.9f} | {g['cosine_median']-bg['cosine_median']:.12f} |")
table='\n'.join(lines)
fig,axs=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
for name,color in [('legacy','#2171b5'),('gain','#d94801'),('total','#238b45')]:
 for ax,key in zip(axs,['cosine_median','positive_dot_fraction']):ax.plot([1.25,1.5,1.75],[d[name][key] for _,d in sets],marker='o',label=name,color=color)
for ax,title in zip(axs,['Median gradient cosine','Positive-dot fraction']):
 ax.set(title=title,xlabel='Common RGB gain',xticks=[1.25,1.5,1.75]);ax.grid(alpha=.2);ax.legend()
fig.suptitle('All 7,346 canonical source states; 1.25 uses frozen T039 evidence')
fig.savefig(root/'high_gain_alignment.png',dpi=170);fig.savefig(root/'high_gain_alignment.pdf');plt.close(fig)
l,g=s['by_gain']['1.75']['legacy'],s['by_gain']['1.75']['gain']
pp=l['positive_dot_fraction']-g['positive_dot_fraction'];cos=l['cosine_median']-g['cosine_median']
text=f"""# T040-A — DONE: {s['classification']}

**SOURCE_REFERENCE_GRADIENT_DIAGNOSTIC_ONLY.** At gain1.75, legacy EV+gamma positive-dot fraction is **{l['positive_dot_fraction']:.12f}**, gain **{g['positive_dot_fraction']:.12f}**, a **{100*pp:.9f} percentage-point** legacy-minus-gain gap. Median cosine is legacy **{l['cosine_median']:.12f}**, gain **{g['cosine_median']:.12f}**, gap **{cos:.12f}**. The fixed scientific criterion requires BOTH gaps>=20pp and>=0.25 at gain1.75 only. Verdict: **{s['classification']}**. Gain1.50 is descriptive transition evidence and creates no second gate.

## Scope, implementation and bindings

Executed source **b47ac253f6019f778dd245887ede89988b589979**, branch `codex/T040A-high-gain-tangent`, PR [#65](https://github.com/word-ky/TTIE/pull/65), base18a55ca1 after accepted T039 merge1a99369f967e4157623749f4df97634180982b44. Evidence/mailbox SHAs are retained in final delivery. Reuse T039 core helpers and accepted Stage-A/Stage-B loops with only two fixed high probes, accepted-selection binding and gain1.75 verdict scope. No deployable code changes, training, optimizer updates, selection changes, gate/renderer/CLIP/energy edits, new cohort, reference-derived grouping or LOL-v2 access.

All **7346 canonical states /400 banks /80 source-training IDs** are reused in the identical accepted T039/T014 manifest order with duplicates and inactive identities retained. No resampling. Accepted selection SHA **{base['selection_sha256']}** is checked before probes; bank entries compare equal. Probe only **1.50 and1.75**, exactly **14692** new states; the completed near-range probes are not rerun. Gain1 renderer reconstruction is reused solely as the accepted parity check, without a new gain1 gradient/reference audit.

T014 source manifest SHA **{f['source_manifest_sha256']}**; canonical bank manifest **{f['bank_manifest_sha256']}**; historical CRLF receipt **{f['source_receipt_sha256']}**; frozen Sobolev checkpoint **{f['checkpoint_sha256']}**. CLIP/prototype identities, normalization buffers and every deployed source hash are retained in freeze/source_binding. All400 banks' bank.pt/bank_images.pt/bank_decisions.json match accepted hashes; original source-supervision files are not opened by Stage A. Normalization exactly reproduces all7346 cached feature statistics. Raw legacy EV/gamma/gates remain exact. Shared gain uses accepted CommonRegion2 raw parameterization exp(log2*tanh), inactive gain1.

Frozen T039 gain1.25 baseline summary SHA **{base['summary_sha256']}**. Gain positive-dot **0.907836644592**, median cosine **0.862191500210** are copied from accepted evidence, along with legacy and total values. No baseline recomputation or refitting occurs. Full copied baseline is in T040A_frozen_baseline.json and Stage-B summary.

## Isolation, execution and verification

Stage A uses only retained source-bank tensor inputs; PIL source-image opening is prohibited. A clean-condition bank is used through its stored input tensor role, never as reference supervision. Gain1 reconstruction maximum pixel error **{f['max_canonical_output_abs_error']} <=1e-6**; cached feature drift **{f['max_canonical_feature_abs_error']}**, recorded under the accepted T039 rule without a feature-equality gate. No cached-feature bit-exact claim.

GPU1 Stage A **20260915-105344-ttie-t040a-stage-a**, release20260915-105308-ttie-t040a-high-gain, completed in **{f['seconds']:.6f}s**. All14692 learned gradients, states, features, energies, masks and output hashes froze **{f['completed_utc']}**, freeze SHA **{b['stage_a_freeze_sha256']}**, source/JPG opens0. Stage B started **{b['started_utc']}**, first source JPG **{b['opened_source_targets'][0]['utc']}**, completed **{b['completed_utc']}**, duration **{b['seconds']:.6f}s**. Stage-B run ID is recorded in run metadata/delivery. Both stages exit0 on A6000 physical GPU1.

Stage B opens only80 authorized source-training JPGs under shared/t008/val2017 using exact accepted T014 load_image: float32 RGB, shorter side320, CPU bicubic antialias and clamp. Isolated reference objective is float64 mean RGB squared error. All **{b['output_hashes_exact']} output hashes match exactly**, all400 Stage-A files unchanged. Zero optimizer updates/selection changes; finite gradients/scalars; raw states and frozen models unchanged. No LOL-v2 low/normal or official-test image is accessed.

Alignment is the unchanged T029/T038 active raw-coordinate convention: legacy EV+gamma, common gain, total. Dot and norms are float64; norm<=1e-12 makes the cosine undefined and excludes that row from cosine/positive fraction. Inactive rows remain in counts and zero/degenerate fractions. Positive dot means a negative learned-energy gradient is locally restorative for RGB MSE, not a finite Adam/projection guarantee.

{table}

![High-gain source tangent alignment](T040A_result/high_gain_alignment.png)

All per-probe scalars, masks reconstructed from gates and raw g_E/g_R vectors are retained in states.json/vectors.json; full means/p10/p90 and energy/reference zero fractions are in summary.json. Independent stdlib replay imports no T029/T038/T039/T040 alignment, summarize or classify helper, checks all bank/state/gain ordering and masks, and recomputes the gain1.75 verdict. **{v['scalar_checks']} scalar checks**, all14692 probes/44076 groups, max absolute error **{v['max_abs_error']} <=1e-6**. Baseline tests2passed; T040 local tests2passed19.31s. Server focused tests2passed1.33s; exact output is retained in Stage-A train.log.

## Failures, limits and archive

Before any GPU experiment, a broad new-worktree checkout filled D. Git sparse-checkout plus restoration of partial checkout files recovered the clean worktree and1.64GB free; no existing project evidence was altered. Automatic approval rejected an initial combined process-stop/removal command, so that removal never ran; Git managed repair succeeded. Two baseline invocations during recovery passed but reported unusually long wall times3026.08s/3038.61s. Preparation therefore exceeded the approximate one-hour budget; experimental scope remained the single fixed audit. Research-lead T040-A-EXEC commit eed82b65 was observed while the same Stage A was running (~58% complete); it accepts the exact executed source and leaves every scientific setting unchanged. No second experiment was launched. The existing NVML warning is nonblocking. No scientific threshold/gain/state change.

Full F archive **{a['archives']['full']['bytes']} bytes**, SHA **{a['archives']['full']['sha256']}**; compact evidence **{a['archives']['compact']['bytes']} bytes**, SHA **{a['archives']['compact']['sha256']}**, both server roots and local. Original T014 bank assets remain at their bound source run. Exact commands/env, state selection, run logs and output receipts are preserved; final Git-exact recovery package mirrored locally and to home/F shared/t040a.

Both legacy and gain alignment decline as common gain increases. Gain positive-dot falls17.729pp and median cosine0.250152 from frozen1.25 to1.75, but the corresponding within1.75 legacy-minus-gain differences remain only6.594923pp/0.158235. This does not meet the prescribed gain-specific deficit gate, and it does not mean the high-gain source field is unchanged or uniformly reliable.

This is a source-training local derivative diagnostic, not a held-out qualification, proof of a deployable repair or proof that retraining would help. T038's real loss-subset and this all-source-state population are different; domain and trajectory effects are not experimentally separated here. Stop and await research-lead judgment regardless of verdict. Do not retrain or add a controller in this cycle.

{s['classification']}
"""
(r/'T040A_report.md').write_text(text,encoding='utf-8');print(s['classification'])
