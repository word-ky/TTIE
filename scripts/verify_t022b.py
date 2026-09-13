"""Independent standard-library aggregation of frozen numeric audit artifacts."""
import csv,json,math,statistics,sys
from collections import Counter
from pathlib import Path


def close(a,b):assert math.isclose(a,b,abs_tol=1e-11,rel_tol=0),(a,b)


def stats(values):
    return dict(mean=statistics.mean(values),median=statistics.median(values),
                p95=statistics.quantiles(values,n=100,method='inclusive')[94])


def main(root,original):
    with (root/'checkpoint_metrics.csv').open() as f:rows=list(csv.DictReader(f))
    for r in rows:
        for k in r:
            if k!='low':r[k]=int(r[k]) if k in ['index','step'] else float(r[k])
    assert len(rows)==4100
    summary=json.loads((root/'summary.json').read_bytes());fixed=json.loads((root/'global_fixed_steps.json').read_bytes())
    freeze=json.loads((original/'freeze.json').read_bytes())
    with (original/'metrics.csv').open() as f:prior=list(csv.DictReader(f))
    grouped={i:[r for r in rows if r['index']==i] for i in range(100)}
    choices={name:[] for name in ['selected','psnr_oracle','ssim_oracle']};original_error=0
    for i,values in grouped.items():
        assert [r['step'] for r in values]==list(range(41))
        selected=values[freeze['rows'][i]['selected_step']]
        for m in ['psnr','ssim']:
            error=abs(selected[m]-float(prior[i]['ours_'+m]));original_error=max(original_error,error)
        choices['selected'].append(selected)
        for m in ['psnr','ssim']:
            choices[m+'_oracle'].append(sorted(values,key=lambda r:(-r[m],r['step']))[0])
    for k in range(41):
        for m in ['psnr','ssim']:
            actual=stats([grouped[i][k][m] for i in range(100)])
            for name,value in actual.items():close(value,fixed[k][m][name])
    for name,values in choices.items():
        assert [Counter(r['step'] for r in values)[k] for k in range(41)]==summary['step_histograms'][name]
        for m in ['psnr','ssim']:
            for k,value in stats([r[m] for r in values]).items():close(value,summary['aggregates'][name][m][k])
        for key,value in summary['saturation'][name].items():close(statistics.mean(r[key] for r in values),value)
    for name in ['psnr_oracle','ssim_oracle']:
        for m in ['psnr','ssim']:
            for k,value in stats([a[m]-b[m] for a,b in zip(choices[name],choices['selected'])]).items():close(value,summary['headroom'][name][m][k])
        close(sum(a['step']==b['step'] for a,b in zip(choices[name],choices['selected']))/100,
              summary['headroom'][name]['selected_equals_oracle_fraction'])
    for m in ['psnr','ssim']:
        best=sorted(range(41),key=lambda k:(-fixed[k][m]['mean'],k))[0]
        assert best==summary['best_global_fixed_step'][m]
    assert all(math.isfinite(r[m]) for r in rows for m in ['psnr','ssim'])
    result=dict(passed=True,rows=4100,images=100,independent='Python statistics/csv, no TTIE/torch/numpy imports',
                fixed_steps_oracles_gains_histograms_saturation_agree=True,selected_vs_original_metric_max_abs_error=original_error)
    (root/'independent_aggregation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main(Path(sys.argv[1]),Path(sys.argv[2]))
