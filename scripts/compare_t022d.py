"""Post-freeze comparison to accepted T022-C; fixed gate and independent checks."""
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import torch


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def accumulate_final(counts,grid,lower,upper,active):
    for ch,channel in enumerate(['ev','gamma']):
        collapsed=lower[0,ch]==upper[0,ch]
        lo=np.abs(grid[0,ch]-lower[0,ch])<=1e-6
        hi=np.abs(grid[0,ch]-upper[0,ch])<=1e-6
        groups={'active':active,'inactive':~active,'collapsed':collapsed,'active_noncollapsed':active & ~collapsed}
        for group,mask in groups.items():
            key=channel+'_'+group
            row=counts.setdefault(key,dict(coordinates=0,lower=0,upper=0,either=0))
            row['coordinates']+=int(mask.sum())
            for name,sat in [('lower',lo),('upper',hi),('either',lo|hi)]:row[name]+=int((sat & mask).sum())


def finish_counts(counts):
    return {k:dict(**v,**{name+'_fraction':v[name]/v['coordinates'] if v['coordinates'] else None for name in ['lower','upper','either']}) for k,v in counts.items()}


def main(current,baseline):
    rows=list(csv.DictReader((current/'metrics.csv').open()))
    old=list(csv.DictReader((baseline/'metrics.csv').open()))
    freeze=json.loads((current/'freeze.json').read_bytes())
    assert len(rows)==len(old)==len(freeze['rows'])==100
    assert freeze['low_only_decoder_enforced']
    assert sha(current/'config.json')==freeze['config_sha256']
    delta_rows=[];saturation=[];final_counts={}
    for r,a,f in zip(rows,old,freeze['rows']):
        assert r['low']==a['low']==f['low'] and r['index']==a['index']
        assert r['raw_psnr']==a['raw_psnr'] and r['raw_ssim']==a['raw_ssim']
        directory=current/f'{f["index"]:03d}'
        for name,receipt in f['files'].items():assert sha(directory/name)==receipt['sha256']
        d=json.loads((directory/'decision.json').read_bytes())
        prior=json.loads((baseline/f'{f["index"]:03d}'/'decision.json').read_bytes())
        assert d['gate']==prior['gate']
        assert d['diagnostics']['steps']==80
        box=d['diagnostics']['action_box'];before=prior['diagnostics']['action_box']
        assert box==before
        assert d['selection']['selected_step']==int(np.argmin(d['selection']['scores']))
        saved=torch.load(directory/'output.pt',map_location='cpu',weights_only=True)
        assert all(torch.isfinite(v).all() for v in saved.values())
        grid=saved['grid'].numpy();lower=np.array(box['lower']);high=np.array(box['upper'])
        sat={}
        for ch,name in enumerate(['ev','gamma']):
            lo=np.abs(grid[0,ch]-lower[0,ch])<=1e-6;hi=np.abs(grid[0,ch]-high[0,ch])<=1e-6
            sat.update({name+'_lower':float(lo.mean()),name+'_upper':float(hi.mean()),name+'_either':float((lo|hi).mean())})
        saturation.append(sat)
        accumulate_final(final_counts,np.array(d['diagnostics']['final_grid']),lower,high,np.array(d['gate']['active']).reshape(2,2))
        delta_rows.append(dict(index=f['index'],low=r['low'],delta_psnr=float(r['ours_psnr'])-float(a['ours_psnr']),
                              delta_ssim=float(r['ours_ssim'])-float(a['ours_ssim']),selected_step=f['selected_step'],**sat))
    with (current/'paired_deltas.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(delta_rows[0]));w.writeheader();w.writerows(delta_rows)
    aggregates={}
    for name,table,prefix in [('raw',rows,'raw'),('T022C',old,'ours'),('T022D',rows,'ours')]:
        aggregates[name]={}
        for metric in ['psnr','ssim']:
            values=[float(r[prefix+'_'+metric]) for r in table]
            assert all(math.isfinite(x) for x in values)
            aggregates[name][metric]=dict(mean=float(np.mean(values)),median=float(np.median(values)))
            assert math.isclose(statistics.mean(values),aggregates[name][metric]['mean'],abs_tol=1e-12,rel_tol=0)
            assert statistics.median(values)==aggregates[name][metric]['median']
    deltas={}
    for metric in ['psnr','ssim']:
        values=[r['delta_'+metric] for r in delta_rows];quantiles=statistics.quantiles(values,n=100,method='inclusive')
        deltas[metric]=dict(mean=float(np.mean(values)),median=float(np.median(values)),p10=float(np.quantile(values,.1)),p90=float(np.quantile(values,.9)))
        for actual,expected in [(statistics.mean(values),deltas[metric]['mean']),(quantiles[9],deltas[metric]['p10']),(quantiles[89],deltas[metric]['p90'])]:
            assert math.isclose(actual,expected,abs_tol=1e-12,rel_tol=0)
    positive=deltas['psnr']['mean']>=.5 and aggregates['T022D']['ssim']['mean']>=aggregates['T022C']['ssim']['mean']
    independent_positive=statistics.mean(r['delta_psnr'] for r in delta_rows)>=.5 and statistics.mean(float(r['ours_ssim']) for r in rows)>=statistics.mean(float(r['ours_ssim']) for r in old)
    assert positive==independent_positive
    times=[float(r['seconds']) for r in rows]
    summary=dict(status='experiment-complete',variant='accepted C with max_steps80 only',rows=100,aggregates=aggregates,paired_deltas=deltas,
                 selected_step_histogram=np.bincount([r['selected_step'] for r in delta_rows],minlength=81).tolist(),
                 selected_step80=sum(r['selected_step']==80 for r in delta_rows),
                 final_saturation=finish_counts(final_counts),
                 runtime_seconds=dict(mean=float(np.mean(times)),median=float(np.median(times)),p95=float(np.quantile(times,.95))),
                 saturation={key:float(np.mean([r[key] for r in saturation])) for key in saturation[0]},
                 gate=dict(psnr_delta_threshold=.5,ssim_must_not_decrease=True,psnr_pass=deltas['psnr']['mean']>=.5,ssim_pass=aggregates['T022D']['ssim']['mean']>=aggregates['T022C']['ssim']['mean']),
                 verdict='materially positive' if positive else 'negative/insufficient')
    (current/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    verification=dict(passed=True,all100_gates_identical_to_C=True,all100_action_boxes_identical_to_C=True,
                      all100_frozen_scientific_artifact_hashes_unchanged_after_reference_evaluation=True,
                      raw_metrics_identical_to_C=True,statistics_numpy_independent_aggregation_and_gate_agree=True)
    (current/'comparison_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps(summary));print(json.dumps(verification))


if __name__=='__main__':main(Path(sys.argv[1]),Path(sys.argv[2]))
