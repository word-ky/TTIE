"""Independent standard-library replay of every published scalar diagnostic."""
import csv,json,math,statistics,sys
from collections import Counter
from pathlib import Path

out=Path(sys.argv[1]);prior=Path(sys.argv[2])
steps=list(csv.DictReader((out/'per_step.csv').open()))
images=list(csv.DictReader((out/'per_image.csv').open()))
old=list(csv.DictReader(prior.open()));summary=json.loads((out/'summary.json').read_bytes())
curves=json.loads((out/'curves.json').read_bytes());freeze=json.loads((out/'freeze.json').read_bytes())
receipt=json.loads((out/'evaluation_receipt.json').read_bytes())
assert len(steps)==8200 and len(images)==len(old)==100
assert freeze['completed_utc']<receipt['started_utc']<=min(r['utc'] for r in receipt['normal_opens'])
errors=[]
def close(a,b):
    errors.append(abs(float(a)-float(b)));assert errors[-1]<1e-10,(a,b)
def statistics_replay(values,s):
    values=sorted(values);n=len(values)
    def quantile(p):
        i=(n-1)*p;lo=math.floor(i);hi=math.ceil(i)
        return values[lo]+(values[hi]-values[lo])*(i-lo)
    for k,v in dict(count=n,mean=statistics.mean(values),median=statistics.median(values),min=values[0],max=values[-1],p05=quantile(.05),p95=quantile(.95)).items():close(v,s[k])
headroom={m:[] for m in ['psnr','ssim']};losses={m:[] for m in headroom};rescues=Counter();all_rescues=Counter();best_steps={m:[] for m in headroom}
for image,accepted,frozen in zip(images,old,freeze['rows']):
    assert image['low']==accepted['low']==frozen['low']
    rows={method:sorted([r for r in steps if r['low']==image['low'] and r['method']==method],key=lambda r:int(r['step'])) for method in ['baseline','common']}
    for method in rows:
        assert [int(r['step']) for r in rows[method]]==list(range(41))
        selected=int(image[method+'_selected_step']);assert selected==int(accepted[method+'_step'])==frozen['methods'][method]['selected_step']
        assert selected==min(range(41),key=lambda i:float(rows[method][i]['learned_energy']))
        for step,r in enumerate(rows[method]):close(r['learned_energy'],frozen['methods'][method]['energies'][step])
        for metric in headroom:close(rows[method][selected][metric],accepted[method+'_'+metric])
    for metric in headroom:
        base=float(accepted['baseline_'+metric]);chosen=int(image['common_selected_step'])
        values=[float(r[metric]) for r in rows['common']];best=max(range(41),key=lambda i:values[i])
        assert best==int(image['common_reference_best_'+metric+'_step']);best_steps[metric].append(str(best))
        delta=values[best]-values[chosen];close(delta,image[metric+'_headroom']);headroom[metric].append(delta)
        close(values[best],image['common_reference_best_'+metric])
        earlier=[i for i in range(chosen) if values[i]>=base]
        assert bool(earlier)==(image['earlier_'+metric+'_reaches_baseline']=='True')
        assert image['earliest_'+metric+'_rescue_step']==(str(earlier[0]) if earlier else '')
        all_rescues[metric]+=bool(earlier)
        loss=float(accepted['delta_'+metric])<0;assert loss==(image['prior_'+metric+'_loss']=='True')
        if loss:losses[metric].append(image['low']);rescues[metric]+=bool(earlier)
for metric,values in headroom.items():
    statistics_replay(values,summary['headroom'][metric]);assert sum(v>0 for v in values)==summary['positive_headroom'][metric]
    assert dict(Counter(best_steps[metric]))==summary['best_step_histograms'][metric]
    assert len(losses[metric])==summary['prior_loss_counts'][metric]
    assert rescues[metric]==summary['earlier_rescued_loss_counts'][metric]
    assert all_rescues[metric]==summary['earlier_reaches_baseline_all'][metric]
for method,curve in curves.items():
    for row in curve:
        selected=[r for r in steps if r['method']==method and int(r['step'])==row['step']]
        for metric in ['psnr','ssim','learned_energy']:statistics_replay([float(r[metric]) for r in selected],row[metric])
verdict='strong late-selection headroom' if statistics.mean(headroom['psnr'])>=.75 and rescues['psnr']>=15 else 'limited/mixed late-selection headroom'
assert verdict==summary['classification']
result=dict(status='PASS',per_step_rows=8200,per_image_rows=100,scalar_checks=len(errors),max_abs_error=max(errors),prior_loss_counts={m:len(v) for m,v in losses.items()},rescues=dict(rescues),classification=verdict,freeze_before_every_reference=True)
(out/'local_scalar_replay.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
