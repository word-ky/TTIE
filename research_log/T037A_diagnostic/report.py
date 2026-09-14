"""Render the diagnostic report and scientific curves from verified scalar evidence."""
from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[2];out=root/'research_log/T037A_result'
s=json.loads((out/'summary.json').read_bytes());c=json.loads((out/'curves.json').read_bytes())
f=json.loads((out/'freeze.json').read_bytes());e=json.loads((out/'evaluation_receipt.json').read_bytes())
v=json.loads((out/'local_scalar_replay.json').read_bytes());r=json.loads((root/'research_log/T037A_reconstruction_receipt.json').read_bytes())
loss=json.loads((out/'prior_loss_cases.json').read_bytes());worst=json.loads((out/'worst_case_trajectory.json').read_bytes())
fig,axes=plt.subplots(2,3,figsize=(13,7.2),constrained_layout=True)
for col,(metric,label) in enumerate([('psnr','PSNR (dB)'),('ssim','RGB-SSIM'),('learned_energy','Learned energy')]):
    for method,color in [('baseline','#3268a8'),('common','#cc6633')]:
        axes[0,col].plot(range(41),[p[metric]['mean'] for p in c[method]],label=method,color=color)
        values=[p for p in worst if p['method']==method]
        axes[1,col].plot([p['step'] for p in values],[p[metric] for p in values],color=color,label=method)
    axes[0,col].set_title('100-image mean');axes[1,col].set_title('Worst prior loss: low00559.png')
    for ax in axes[:,col]:ax.set_xlabel('Frozen step');ax.set_ylabel(label);ax.grid(alpha=.2);ax.legend()
fig.suptitle('T037-A — reference diagnostic only; original selections unchanged')
fig.savefig(out/'trajectory_curves.png',dpi=160);fig.savefig(out/'trajectory_curves.pdf');plt.close(fig)
h=s['headroom'];w=s['worst_case']
text=f'''# T037-A — DONE: {s['classification']}

**REFERENCE_DIAGNOSTIC_ONLY.** Within the exact frozen common-gain trajectories, mean reference-best minus selected PSNR is **{h['psnr']['mean']:.12f} dB**; **{s['earlier_rescued_loss_counts']['psnr']}/29** prior PSNR-loss images reach or exceed the baseline selected PSNR at a strictly earlier common step. The predefined joint rule is mean headroom >=0.75 dB AND at least15/29 earlier rescues. This audit does not select a deployable checkpoint or establish an early-stopping rule.

Late overshoot is visible in a subset, especially the worst prior PSNR case, but does not explain the entire tail:11/29 prior PSNR-loss images have no earlier common state reaching the baseline selected PSNR. Median PSNR headroom is only0.091785222 dB, while p95 is4.434495354 dB; the mean is driven disproportionately by a smaller tail. The worst case can recover PSNR but its best common SSIM still falls below baseline. The fixed joint criterion therefore remains unmet; neither the threshold nor the cohort is changed.

## Inputs and information boundary

Source **009823f96f987d341d16314e924d0e3cd24b3b9e**, branch `codex/T037A-late-selection`, PR [#62](https://github.com/word-ky/TTIE/pull/62), base b7872a7a after accepted T036 merge d7e615066479fb97329aec1d28df636015433dd8. Evidence SHA is in the final main mailbox/delivery receipt.

Exact T036 cohort SHA **{f['cohort_sha256']}**, prior freeze **{f['prior_freeze_sha256']}**, prior metrics **{f['prior_metrics_sha256']}**. Reuses those same100 already-reference-used development pairs. No new cohort, official-test access, optimizer update, energy forward pass, energy/CLIP training, deployable-state mutation or stopping threshold. Original min-predicted-energy decisions and all41 energy scores are copied unchanged. Reference-best states are offline diagnostics only and must never become per-image inference inputs.

T036 retained raw-state trajectories, physical fields and200 selected output tensors, rather than all step images. The accepted renderer sources were deployed as exact accepted Git bytes and bound by127 file hashes. A6000 physicalGPU1 reconstructed all8,200 images from fixed lows, raw states and gates, with no optimizer or energy model instantiated. Every prior artifact hash was checked. All200 selected outputs are **bit-exact**, maximum absolute error **{r['selected_max_abs']}**; physical-field replay max **{r['physical_field_max_abs']}**, identity max **{r['identity_max_abs']}**, both within the predeclared1e-6 reconstruction tolerance. Small field/identity differences reflect floating-point rendering; no selected-metric discrepancy was introduced.

Sole GPU reconstruction `20260915-055148-ttie-t037a-reconstruct`, release `20260915-055124-ttie-t037a-diagnostic`, completed in **{f['seconds']:.6f}s**, exit0. All200 complete trajectory tensors froze at **{f['completed_utc']}**, SHA **{r['freeze_sha256']}**, before references; normal decodes0 and updates0. Evaluation `20260915-055515-ttie-t037a-evaluate` began **{e['started_utc']}**, first normal open **{min(p['utc'] for p in e['normal_opens'])}**, completed **{e['completed_utc']}**, in **{e['seconds']:.6f}s**, exit0. Every full tensor is hashed again before its reference is opened. All200 hashes are checked again for backup after scoring.

## Numerical verification

Exact accepted native full-RGB PSNR/RGB-SSIM, float32 pixels promoted to float64, no crop/resize/quantized reconstruction. Eight CPU workers reuse the accepted SciPy Gaussian-filter SSIM implementation; an independent separable-filter implementation and independent dot-product MSE replay every8,200 metric pair. Maximum metric discrepancy **{e['independent_metric_max_abs_error']}**. All200 selected T036 PSNR/SSIM values reproduce with max error **{e['selected_metric_max_abs_error']}**. Original selected steps reproduce exactly, including99 common step40 and one step36. Local standard-library replay verifies all8,200 rows,100 per-image diagnostics, original energy selections, histograms, loss IDs, quantiles and curves: **{v['scalar_checks']} scalar checks**, max error **{v['max_abs_error']}**. Focused tests: local2passed1.95s; server2passed0.10s.

## Headroom and rescue

| Metric | Mean headroom | Median | p05 | p95 | Positive images | Earlier-rescued prior losses | Earlier reaches baseline, all100 |
|---|---:|---:|---:|---:|---:|---:|---:|
'''
for m in ['psnr','ssim']:
    z=h[m];text+=f"| {m} | {z['mean']:.12f} | {z['median']:.12f} | {z['p05']:.12f} | {z['p95']:.12f} | {s['positive_headroom'][m]} | {s['earlier_rescued_loss_counts'][m]}/{s['prior_loss_counts'][m]} | {s['earlier_reaches_baseline_all'][m]} |\n"
text+='\nReference-best common-step histograms (earliest exact ties; PSNR and SSIM optimized independently):\n\n'
for m in ['psnr','ssim']:text+=f"- {m}: "+', '.join(f"{k}: {count}" for k,count in sorted(s['best_step_histograms'][m].items(),key=lambda p:int(p[0])))+'.\n'
text+='\n## Worst prior case: low00559.png\n\n'
for m in ['psnr','ssim']:
    text+=f"{m}: baseline selected step{w['baseline_selected_step']} = {w['baseline_selected_'+m]:.12f}; common selected step{w['common_selected_step']} = {w['common_selected_'+m]:.12f}; common reference-best step{w['common_reference_best_'+m+'_step']} = {w['common_reference_best_'+m]:.12f}; headroom {w[m+'_headroom']:.12f}; earliest baseline-reaching common step = {w['earliest_'+m+'_rescue_step']}.\n\n"
text+='Full82-point worst-case trajectory is in `T037A_result/worst_case_trajectory.json`; plotted with aggregate curves below.\n\n![Frozen trajectory curves](T037A_result/trajectory_curves.png)\n\n## Exact29 prior PSNR-loss cases\n\n| Image | Baseline selected PSNR | Common selected PSNR | Reference-best step | Best PSNR | Headroom | Earliest earlier rescue |\n|---|---:|---:|---:|---:|---:|---:|\n'
for z in loss['psnr']:
    text+=f"| {z['low']} | {z['baseline_selected_psnr']:.9f} | {z['common_selected_psnr']:.9f} | {z['common_reference_best_psnr_step']} | {z['common_reference_best_psnr']:.9f} | {z['psnr_headroom']:.9f} | {z['earliest_psnr_rescue_step']} |\n"
text+='\nAll40 prior SSIM-loss cases and their independent SSIM rescue results are in `T037A_result/prior_loss_cases.json`; all100 per-image diagnostics in `per_image.csv`.\n\n## All per-step aggregate curves\n\nEach quality/energy cell is mean / median over the same100 images. Learned energies are copied from original T036 decisions, never recomputed or optimized.\n\n| Method | Step | PSNR | RGB-SSIM | Learned energy |\n|---|---:|---:|---:|---:|\n'
for method,curve in c.items():
    for z in curve:text+=f"| {method} | {z['step']} | {z['psnr']['mean']:.9f} / {z['psnr']['median']:.9f} | {z['ssim']['mean']:.9f} / {z['ssim']['median']:.9f} | {z['learned_energy']['mean']:.9f} / {z['learned_energy']['median']:.9f} |\n"
text+='''
## Commands, artifacts and deviations

`reconstruct.py --accepted <T036 audit> --manifest research_log/T036A_cohort/manifest.json --low-root <shared/t036a/low> --binding research_log/T037A_source_binding.json --out <run/artifacts/REFERENCE_DIAGNOSTIC_ONLY>` after `python -m pytest -q tests/test_t037a_diagnostic.py`; then separate `evaluate.py --accepted <T036 audit> --manifest <same manifest> --normal-root <shared/t036a/normal> --out <frozen output>`. Local `replay_scalars.py research_log/T037A_result research_log/T036A_result/audit/metrics.csv`.

The initial evaluation launch055422 had an SSH connection timeout before any remote run directory/session existed; absence was checked, then sole actual evaluation055515 launched. No reconstruction/evaluation scientific restart or retuning. The existing NVML warning did not prevent A6000 CUDA execution. No changes to accepted deployable modules. Full reconstructed floating-point frames remain under the reconstruction run and in an F archive; compact scalar evidence and source/recovery files are retained under both server roots and locally. Archive SHA256 values are in `T037A_archives.json`; recovery/delivery identifies the final Git commits.

Stop after this diagnostic. Await the research lead's review; no threshold/controller qualification is performed in this cycle. Reachable reference-best quality is diagnostic capacity, not deployable performance.

'''+s['classification']+'\n'
(root/'research_log/T037A_report.md').write_text(text,encoding='utf-8')
print(s['classification'])
