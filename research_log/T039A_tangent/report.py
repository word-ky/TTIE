"""Generate the source-only diagnostic report from independently verified evidence."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[2];out=root/'research_log/T039A_result'
f=json.loads((out/'stage_a/freeze.json').read_bytes());r=json.loads((out/'stage_b/receipt.json').read_bytes());s=json.loads((out/'stage_b/summary.json').read_bytes());v=json.loads((out/'independent_replay.json').read_bytes());a=json.loads((root/'research_log/T039A_archives.json').read_bytes());selection=json.loads((out/'stage_a/selection.json').read_bytes())
l,g=s['overall']['legacy'],s['overall']['gain'];fraction_gap=l['positive_dot_fraction']-g['positive_dot_fraction'];cos_gap=l['cosine_median']-g['cosine_median']
fig,axes=plt.subplots(1,2,figsize=(10.5,4),constrained_layout=True)
gains=[.75,1.,1.25]
for group,color in [('legacy','#3268a8'),('gain','#cc6633'),('total','#438c59')]:
    for ax,key in zip(axes,['cosine_median','positive_dot_fraction']):ax.plot(gains,[s['by_gain'][str(gain)][group][key] for gain in gains],marker='o',label=group,color=color)
for ax,label in zip(axes,['Median cosine','Positive-dot fraction']):ax.set_xlabel('Fixed common gain');ax.set_ylabel(label);ax.set_xticks(gains);ax.grid(alpha=.2);ax.legend()
fig.suptitle('T039-A — source reference diagnostic only; all7346 canonical states')
fig.savefig(out/'source_gain_alignment.png',dpi=170);fig.savefig(out/'source_gain_alignment.pdf');plt.close(fig)
text=f'''# T039-A — DONE: {s['classification']}

**SOURCE_REFERENCE_GRADIENT_DIAGNOSTIC_ONLY.** Across all22,038 fixed probes from the7,346 accepted canonical T014 source-training states, legacy positive-dot fraction is **{l['positive_dot_fraction']:.12f}**, gain **{g['positive_dot_fraction']:.12f}**; legacy-minus-gain gap **{100*fraction_gap:.9f} percentage points**. Median cosine is legacy **{l['cosine_median']:.12f}**, gain **{g['cosine_median']:.12f}**, difference **{cos_gap:.12f}**. The predefined deficit criterion requires both gaps>=20 percentage points and>=0.25 respectively; the resulting classification is **{s['classification']}**. No thresholds were fitted to these results.

This is a local derivative diagnosis on already-authorized source-training data, not a held-out qualification or a retraining result. It does not establish a deployable gain controller or prove that source retraining will repair the real-domain tail. T038's real-domain loss-subset association and this all-source-state diagnostic refer to different populations; no target-domain image or result entered this audit's inputs or grouping.

## Provenance and deterministic state selection

Executed source **04df2ec74911f5fbf04724f5031560da730e59d1**, branch `codex/T039A-source-gain-tangent`, PR [#64](https://github.com/word-ky/TTIE/pull/64), baseed0acc09 after T038 accepted merge689798b5ab789bfd9133bad9a1584118725e3421. Evidence/main-mailbox SHAs are recorded in the final delivery receipt.

Accepted T014 source manifest SHA **{f['source_manifest_sha256']}**; T014/T031 source bank manifest **{f['bank_manifest_sha256']}**; T031-bound T014 receipt **{f['source_receipt_sha256']}**; frozen Sobolev checkpoint **{f['checkpoint_sha256']}**. CLIP checkpoint **{f['model_identity']['sha256']}**, prototypes **{f['prototype_sha256']}**. The exact gate/calibration comes from the T014 frozen receipt, not the later real-domain gate. All400 bank manifests and bank.pt/bank_images.pt/bank_decisions.json hashes were checked before probe computation. Source supervision files/target labels were not loaded by Stage A. Seven directly reused T014 modules (renderer, semantic objective, energy, normalization head, natural image loading, CLIP, prototypes) match their accepted source hashes exactly; current CommonRegion2 and the unchanged energy-feature evaluator are accepted donor code. All deployed files are source-bound in the freeze.

Use **all7346** canonical rows from **400** banks over the exact **80 train_t014_sobolev IDs** already used by T014/T031, in accepted manifest order. Retain the original bank names/order, duplicates and no-active identity rows. No metric-, gradient-, label- or reconstruction-quality-based sampling occurred. The full selection and IDs were written before the first probe gradient, selection SHA **{f['selection_sha256']}**. Each row receives exactly **0.75,1.00,1.25**, yielding **22038** probes. Each active Region2 cell receives the specified RGB-shared post-gamma gain; inactive cells keep gain1. Raw legacy EV/gamma and accepted active gate remain fixed. This is not a sweep to select a gain: every prescribed probe is reported.

## Reconstruction and reference boundary

T014 retained source-bank images and raw states. Stage A reads these bound tensors only, taking each bank's saved identity image as the source input; it never regenerates a degradation by opening the clean JPG. A bank with condition `clean` is still used solely through its previously retained input tensor, not through reference supervision. Before probing each canonical state, gain1 output parity against the accepted bank image must be<=1e-6. Maximum full-audit canonical pixel difference **{f['max_canonical_output_abs_error']}**. All raw EV/gamma coordinates and gates remain exact. Cached CLIP feature differences are recorded separately: max **{f['max_canonical_feature_abs_error']}**. Reconstruction from the saved identity tensor can introduce small floating-point image/feature differences; it is not claimed to reproduce every cached feature bit-exactly.

The T014 checkpoint's mean/scale normalization buffers were reproduced exactly from all7346 original stored features (float64 population mean/std cast to saved float32), including constant-coordinate scale handling. Frozen energy/scorer tensors and checkpoint files remain unchanged; `.grad` fields stay empty. Each probe stores raw state, learned gradient, feature vector, energy, masks and output SHA256. Full redundant probe output tensors are not saved: Stage B reconstructs and checks every output hash bit-exactly. Input tensors and all gradient/state/feature records remain retained in the full archive.

Complete Stage A GPU1 run **20260915-083056-ttie-t039a-stage-a**, release20260915-083021-ttie-t039a-pixel-parity, took **{f['seconds']:.6f}s**, exit0. All22,038 learned gradients/output hashes froze at **{f['completed_utc']}**, SHA **{r['stage_a_freeze_sha256']}**, with source-target/JPG opens0. Stage B began **{r['started_utc']}**; first source reference **{min(z['utc'] for z in r['opened_source_targets'])}**; completed **{r['completed_utc']}**, **{r['seconds']:.6f}s**, exit0. Its run ID is retained in `stage_b_run/meta.json` and the delivery receipt.

Only the80 manifest-pinned training JPGs are permitted in Stage B. It uses the exact accepted T014 `load_image`: RGB float32, shorter side320, bicubic/antialias resize on CPU and clamp. The isolated reference gradient differentiates mean squared RGB error with float64 accumulation at each frozen state. Every one of **22038 output hashes matches** Stage A, and all400 Stage-A files remain unchanged after gradient work and archiving. Zero optimizer updates, zero selection changes, all raw states unchanged during gradient calls, all finite gradients/scalars (undefined degenerate cosines are null under the accepted convention). No LOL-v2 low/normal/test image or external baseline is accessed.

## Alignment results

Exactly the T029 convention: active raw coordinates only; legacy channelsEV/gamma, gain the single shared-gain raw channel, total their disjoint union. Compute float64 dot and Euclidean norms, cosine dot/(norm product), with norm<=1e-12 treated as degenerate. Median cosine and positive-dot fraction exclude degenerate rows per group; zero/degenerate fractions and denominators are explicitly reported. A positive dot means the negative learned-energy gradient is locally restorative for RGB MSE. This is not an Adam/projection/update evaluation.

| Set | Group | All / nondegenerate | Median cosine | Positive-dot fraction | Median energy norm | Median reference norm | Median dot | Degenerate fraction |
|---|---|---:|---:|---:|---:|---:|---:|---:|
'''
for name,groups in [('overall',s['overall'])]+[(f'gain{gain}',s['by_gain'][str(gain)]) for gain in gains]:
    for group,z in groups.items():text+=f"| {name} | {group} | {z['count']} / {z['nondegenerate']} | {z['cosine_median']:.12f} | {z['positive_dot_fraction']:.12f} | {z['energy_norm_median']:.12g} | {z['reference_norm_median']:.12g} | {z['dot_median']:.12g} | {z['either_zero_fraction']:.12f} |\n"
text+=f'''
![Source gain tangent alignment](T039A_result/source_gain_alignment.png)

Full per-probe dots/cosines/norms/signs, means/p10/p90 and group-specific energy/reference zero fractions are retained in `stage_b/states.json` and `summary.json`; exact saved gradient vectors/gates are in `vectors.json`. The independent verifier reconstructs active masks from gates, checks all canonical/probe identities and recomputes all scalars, summaries and the exact verdict without any T029/T038/T039 helper import. **{v['scalar_checks']} checks**, all22038 probes/66114 group vectors, maximum absolute error **{v['max_abs_error']}** <=1e-6. Local tests2passed8.26s and server2passed1.37s verify fixed physical gains, unchanged legacy coordinates, inactive gain identity, disjoint masks and both scientific conditions.

## Failures and deviations

Initial run082440 stopped at metadata checks before bank input/gradient access: T031 bound the CRLF representation of T014 receipt (d437...), whereas Git stores LF (c6c611aa5769d9f857ff57675e745abe8415721c0315458b85d30e1b5b5c9a68). Exact LF-to-CRLF conversion reproduces the already-accepted hash with identical parsed JSON. That historical representation is retained separately as `T039A_bound_T014_receipt.json`; no accepted metadata content was changed. Source manifest, bank manifest and checkpoint already matched.

Second run082640 stopped after7banks/228 learned-gradient probes at an additional cached-feature<=1e-5 threshold introduced by Codex, not required by T039. Low-only examination of all24 states in bank007 found pixel differences<=1.1920928955078125e-7 but CLIP-feature drift up to2.0444393157958984e-5. The extra feature gate was removed; the original pixel<=1e-6/raw/gate/source reconstruction checks, all states/probes and scientific classification criteria were retained unchanged. All feature drift is disclosed above. No reference JPG or reference gradient had been accessed in either incomplete attempt. The complete Stage A restarted once after this change; those incomplete artifacts and logs remain preserved. This is a procedural deviation, not hidden or represented as a single uninterrupted Stage A. Source data and scientific thresholds were not retuned.

The existing NVML warning is nonblocking; complete Stage A and Stage B used A6000 physicalGPU1. There was no optimizer/training/selection run. Full task archive (including incomplete attempts) **{a['archives']['full']['bytes']} bytes**, SHA **{a['archives']['full']['sha256']}** on F; compact evidence **{a['archives']['compact']['bytes']} bytes**, SHA **{a['archives']['compact']['sha256']}**, on both server roots and locally. Original accepted T014 source-bank tensors remain at their bound source run. Final source/recovery package and reports are mirrored in project `research_log/`.

Commands: `stage_a.py --bank-root <accepted T014 repaired audit> --receipt research_log/T039A_bound_T014_receipt.json --manifest research_log/T014_source_manifest.json --checkpoint <T014_energy.pt> --prototypes <accepted prototypes.pt> --binding research_log/T039A_source_binding.json --out <stage_a>`; only after the complete freeze, `stage_b.py --stage-a <frozen stage_a> --manifest <same source manifest> --images <COCO val2017 source folder> --out <stage_b>`; `replay.py research_log/T039A_result`. Exact commands/environment and all attempt logs are retained in run metadata.

Stop after this source-domain diagnostic and await research-lead review. No retraining, deployable change, further probe value, new cohort or real-domain benchmark is performed in this cycle.

{s['classification']}
'''
(root/'research_log/T039A_report.md').write_text(text,encoding='utf-8');print(s['classification'])
