"""REFERENCE_ORACLE_ONLY independent aggregation; never imports the optimizer."""
import csv,json,math,statistics,sys,hashlib
import numpy as np
from collections import Counter
from pathlib import Path
LABEL='REFERENCE_ORACLE_ONLY'
def stats(values):
    a=sorted(values)
    def q(p):
        x=(len(a)-1)*p; i=int(x);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(x-i)
    return dict(mean=statistics.mean(a),median=statistics.median(a),p10=q(.1),p90=q(.9))
def main(root):
    rows=list(csv.DictReader((root/'per_image.csv').open()));assert len(rows)==100
    js=json.loads((root/'per_image.json').read_text())['rows'];assert len(js)==100
    for i,row in enumerate(rows):
        assert row['label']==LABEL and int(row['index'])==i
        for key,val in js[i].items():
            if isinstance(val,(int,float)):assert float(row[key])==val
            else:assert row[key]==val
        assert int(row['identity_updates'])==int(row['selected_updates'])==500
        assert float(row['oracle_mse'])<=float(row['selected_mse'])+1e-10
        assert float(row['reproduction_max_abs'])<=1e-6 and float(row['grid_max_abs'])<=1e-6
        for key in ['raw_psnr','selected_psnr','oracle_psnr','raw_ssim','selected_ssim','oracle_ssim','delta_psnr','delta_ssim']:
            assert math.isfinite(float(row[key]))
        for m in ['psnr','ssim']:assert abs(float(row['delta_'+m])-(float(row['oracle_'+m])-float(row['selected_'+m])))<1e-12
    aggregates={name:{m:stats([float(r[name+'_'+m]) for r in rows]) for m in ['psnr','ssim']} for name in ['raw','selected','oracle']}
    deltas={m:stats([float(r['delta_'+m]) for r in rows]) for m in ['psnr','ssim']}
    hist={key:dict(sorted(Counter(int(r[key]) for r in rows).items())) for key in ['best_step','identity_best_step','selected_best_step']}
    hist['final_step_all_starts']={500:200}
    active=sum(int(r['active_count']) for r in rows)
    saturation={}
    for key in rows[0]:
        if key.endswith('_inactive_count') and key.startswith(('ev_','gamma_')):
            count=sum(int(r[key]) for r in rows);saturation[key]=dict(count=count,denominator=400-active,fraction=count/(400-active))
        if key.endswith('_active_count') and key.startswith(('ev_','gamma_')):
            count=sum(int(r[key]) for r in rows);saturation[key]=dict(count=count,denominator=active,fraction=count/active)
        if key.endswith('_all_count'):
            count=sum(int(r[key]) for r in rows);saturation[key]=dict(count=count,denominator=400,fraction=count/400)
    summary=dict(label=LABEL,status='oracle ceiling measured',pairs=100,aggregates=aggregates,paired_oracle_minus_T026A=deltas,start_wins=dict(Counter(r['winner'] for r in rows)),step_histograms=hist,saturation=saturation,active_regions=active,inactive_regions=400-active,images_ssim_worse=sum(float(r['delta_ssim'])<0 for r in rows),images_psnr_improved=sum(float(r['delta_psnr'])>1e-6 for r in rows),reproduction_max_abs=max(float(r['reproduction_max_abs']) for r in rows),runtime_seconds=stats([float(r['seconds']) for r in rows]),interpretation='Fixed two-start local oracle search measures demonstrated reachability, not a certified global optimum or deployable enhancement result. MSE optimization need not improve SSIM on every image.')
    mean=deltas['psnr']['mean'];median=deltas['psnr']['median']
    summary['classification']='substantial within-family headroom' if mean>=2 and median>=1 else ('limited within-family headroom' if mean<1 else 'mixed within-family headroom')
    for metric in ['psnr','ssim']:
        values=np.array([float(r['delta_'+metric]) for r in rows])
        check=dict(mean=float(values.mean()),median=float(np.median(values)),p10=float(np.quantile(values,.1)),p90=float(np.quantile(values,.9)))
        assert max(abs(check[k]-deltas[metric][k]) for k in check)<1e-12
    (root/'independent_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    receipt=dict(label=LABEL,passed=True,rows=100,csv_json_identical=True,all_200_starts_500_updates=True,all_mse_nonworse=True,all_reproduction_tolerances_pass=True,optimizer_imported=False,aggregation='Python statistics and independently implemented linear quantile from CSV',csv_sha256=hashlib.sha256((root/'per_image.csv').read_bytes()).hexdigest())
    (root/'independent_aggregation.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':main(Path(sys.argv[1]))
