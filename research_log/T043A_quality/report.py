"""Render only the verified frozen-output quality results."""
from pathlib import Path
import json,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path('research_log');root=r/'T043A_result'
s=json.loads((root/'summary.json').read_bytes());b=json.loads((root/'output_binding.json').read_bytes());e=json.loads((root/'receipt.json').read_bytes());v=json.loads((root/'independent_replay.json').read_bytes());a=json.loads((r/'T043A_archives.json').read_bytes());rows=list(csv.DictReader((root/'metrics.csv').open()))
assert v['status']=='PASS' and v['classification']==s['classification']
m=s['means'];stats=s['paired']
table='| Metric | Baseline mean | Candidate mean | Paired mean | Median | p10 | p90 | Positive / negative / zero |\n|---|---:|---:|---:|---:|---:|---:|---:|'
for metric,d in stats.items():table+=f"\n| {metric} | {m['baseline_'+metric]:.12f} | {m['candidate_'+metric]:.12f} | {d['mean']:.12f} | {d['median']:.12f} | {d['p10']:.12f} | {d['p90']:.12f} | {d['positive']} / {d['negative']} / {d['zero']} |"
cases=['| Metric | Tail | Image | Candidate-minus-baseline |','|---|---|---|---:|']
for metric,d in stats.items():
 for tail in ['worst','best']:
  for x in d[tail]:cases.append(f"| {metric} | {tail} | {x['low']} | {x['delta']:.12f} |")
cases='\n'.join(cases)
fig,axs=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
for ax,metric,title in zip(axs,['psnr','ssim'],['Paired PSNR change (dB)','Paired RGB-SSIM change']):
 vals=sorted(float(x['delta_'+metric]) for x in rows);ax.bar(range(100),vals,color=['#d94801' if x<0 else '#238b45' for x in vals],width=1);ax.axhline(0,color='black',linewidth=.7);ax.set(title=title,xlabel='Matched pairs sorted by metric change',xlim=(-1,100));ax.grid(axis='y',alpha=.2)
fig.suptitle('Step-10 legacy minus selected legacy; both frozen at gain 1.75')
fig.savefig(root/'quality_deltas.png',dpi=170);fig.savefig(root/'quality_deltas.pdf');plt.close(fig)
text=f"""# T043-A — DONE: {s['classification']}

**REFERENCE_EVALUATION_ONLY.** At the exact matched gain1.75, T042 step10-legacy outputs are worse on average than T041 selected-legacy outputs: paired **ΔPSNR={stats['psnr']['mean']:.12f} dB**, **ΔRGB-SSIM={stats['ssim']['mean']:.12f}**. The sole gate requires meanΔPSNR>=+0.50dB AND meanΔRGB-SSIM>=0; neither condition passes. Verdict: **{s['classification']}**. Every one of the100 matched pairs is retained; no thresholds or per-image choices were adjusted.

T042's restored local gradient alignment does not establish better quality at the substituted state. The original selected state can have lower reference error while its learned local direction is less restorative. This result blocks the proposed quality bridge for this fixed substitution; it does not negate T042's local-field measurement. No controller, global step10 checkpoint or inference change is justified by this audit alone.

## Frozen inputs and information boundary

Executed/tested source **bacb3dbc7f1f465d84fb8c2ff436fff3678441f7**, branch `codex/T043A-matched-gain-quality`, PR [#68](https://github.com/word-ky/TTIE/pull/68), base12b89ca1 after accepted T042 mergec03e1642b225575fa6c28dd4c8bf77333c348d7c. Evidence/main-mailbox SHAs are in final delivery. The accepted `ttie/ssim_transfer.py` implementation is reused unchanged; its source hash and every evaluation/test/helper/cohort binding are recorded in output_binding.json.

Exact T036100 cohort SHA **{b['cohort_sha256']}**. Baseline is the retained T041 selected-legacy+gain1.75 output at fixed index1 in Stage-A run20260915-122826; candidate is retained T042 step10-legacy+gain1.75 at index0 in run20260915-133013. These100 normal pairs were already reference-used by T041/T042; no fresh cohort or official-test image is accessed.

- T041 accepted freeze SHA **{b['freeze_sha256']['baseline']}**.
- T042 accepted freeze SHA **{b['freeze_sha256']['candidate']}**.

Both accepted per-image tensor-file hashes, names/order, output indices, native shapes, raw-state hashes and output hashes are verified before reference access. Active gain raw values equal the accepted1.75 conversion exactly and inactive gain is identity. Both gates/assets are equal; candidate legacy-step metadata is10 and the raw legacy hash matches its frozen provenance, while baseline selected-step metadata matches T041. All200 native output tensors are loaded directly and cloned only for read-only scoring; **no rendering, adaptation or selection is run**.

All100 matched pair identities were persisted in output_binding.json at **{b['completed_utc']}**, SHA **{e['output_binding_sha256']}**, with **normal opens0**. Image.open is prohibited during all output binding. Only afterward is a normal-only allowlist enabled; first normal decode **{min(x['utc'] for x in e['opened_normals'])}**. Each normal hash is checked against the accepted cohort before decode. All200 cached output hashes and original tensor-file hashes are checked unchanged after scoring and again at archive. Zero optimizer updates, state changes, selection changes, per-image output choices or test access.

## Exact evaluation convention and results

Use the accepted T026/T036 native400x600 RGB convention: outputs are retained float32[0,1] tensors, cast to float64; normal RGB uint8 is divided by255 in float64, rounded to float32 then cast to float64. No resizing, crop, luminance-only conversion, reference brightness matching or normalization. Per-image PSNR is -10log10 of full RGB mean squared error. RGB-SSIM uses an11x11 Gaussian window sigma1.5, population covariance, K1=.01/K2=.03, data_range1, half-sample symmetric reflect padding, all image borders retained and mean over RGB; all arithmetic float64.

{table}

Quantiles are NumPy linear interpolation, paired means/medians use100 candidate-minus-baseline differences. Positive/negative/zero use exact sign, without a tolerance threshold. Tail cases are descriptive only; none is used to change outputs or the primary gate.

![Frozen matched-gain paired quality changes](T043A_result/quality_deltas.png)

{cases}

The largest PSNR recovery is low00559 (+7.117375dB), but the aggregate and median are negative and85/100 pairs lose PSNR. This is not a globally beneficial early-state substitution. No original variable-gain T036 state is scored or substituted into the matched comparison.

## Independent verification and execution

Run **20260915-135922-ttie-t043a-quality**, release20260915-135847-ttie-t043a-quality, completed exit0 on the A6000 server. This is CPU evaluation using the unchanged accepted SciPy metric, with no model or adaptation work. Main scoring including binding took **{e['seconds']:.6f}s**; independent replay **{v['seconds']:.6f}s**. Baseline SSIM cross-check against the full independent skimage SSIM map passed1test1.44s; T0432local tests passed0.17s and2server tests0.09s.

Independent replay separately reloads both frozen tensors and the same normal for every pair, checks tensor-file/output/normal hashes, recomputes PSNR via torch float64 reduction/log10 and RGB-SSIM via explicit11-tap separable scipy.convolve1d weights, without calling the main T043 metric/aggregation/verdict helpers. It recomputes all paired deltas, population means, means/medians/p10/p90/sign counts, best/worst cases and the single gate using standard-library statistics. **100 pairs /400 output-metric evaluations /{v['scalar_checks']} checked scalars**, max absolute metric/aggregate discrepancy **{v['max_abs_error']} <=1e-10**, PASS. The independent replay itself verifies the pre-reference binding timestamp. Source/checkpoint selection is never recomputed.

## Failures, artifacts and next step

No failures, restarts, deviations or scientific changes; one main evaluation and one independent replay. All metrics finite. No GPU metric replacement was introduced because exact reuse of the accepted evaluation implementation was the requirement. No optimizer, renderer, CLIP/energy call, policy, training, fresh cohort or official-test access.

Evidence archive **{a['bytes']} bytes**, SHA **{a['sha256']}**, verified under home/F shared/t043a and locally. It contains pair-output binding, main/independent receipts, per-image metrics, summary and run logs. Original immutable tensor bundles remain at their bound T041/T042 Stage-A paths, with existing F backups `shared/t041a/T041A_full.tar` and `shared/t042a/T042A_full.tar`; their200 original tensor files were reverified unchanged. Final Git-exact source/evidence/mailbox recovery and reports/plots are mirrored under project research_log and both server roots. Exact evaluation/replay commands/environment are in run/run.sh.

Stop after this evaluation bridge. Recommend research-lead reassessment of the distinction between current-state restoration error and the learned direction before choosing an intervention. Do not implement a stopping/freezing controller from T042 alone, rerun adaptation, or seek a different step/gain in this cycle.

{s['classification']}
"""
(r/'T043A_report.md').write_text(text,encoding='utf-8');print(s['classification'])
