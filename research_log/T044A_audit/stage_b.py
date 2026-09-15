"""Attach only the accepted paired PSNR table, after all scores are frozen."""
from core import *
from association import summarize
import argparse,csv,time
p=argparse.ArgumentParser()
for k in ['stage-a','metrics','out']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();start=time.perf_counter();f=json.loads((a.stage_a/'freeze.json').read_bytes())
assert sha(a.stage_a/'scores.json')==f['scores_sha256'] and sha(a.stage_a/'states.pt')==f['states_sha256']
rows=json.loads((a.stage_a/'scores.json').read_bytes());assert len(rows)==f['count']==100
attached_utc=utc();assert f['completed_utc']<attached_utc
assert sha(a.metrics)=='cdbd7fec76194153db645133d5d35dd43f6f9d852ebbd6674756d77d1ad7aee0'
metrics=list(csv.DictReader(a.metrics.open()));assert len(metrics)==100
for r,m in zip(rows,metrics):
    assert int(m['index'])==r['index'] and m['low']==r['low'] and int(m['common_step'])==r['selected_step']
    delta=float(m['common_psnr'])-float(m['baseline_psnr']);assert delta==float(m['delta_psnr'])
    r.update(delta_psnr=delta,loss=delta<0)
s=summarize([r['D_legacy'] for r in rows],[r['delta_psnr'] for r in rows]);assert s['loss_count']==29 and s['non_loss_count']==71
s['smallest']=sorted(rows,key=lambda r:(r['D_legacy'],r['index']))[:5];s['largest']=sorted(rows,key=lambda r:(-r['D_legacy'],r['index']))[:5]
a.out.mkdir(parents=True,exist_ok=False);write(a.out/'pairs.json',rows);write(a.out/'summary.json',s)
write(a.out/'receipt.json',dict(completed_utc=utc(),metric_open_utc=attached_utc,score_freeze_utc=f['completed_utc'],stage_a_freeze_sha256=sha(a.stage_a/'freeze.json'),scores_sha256=sha(a.stage_a/'scores.json'),metric_sha256=sha(a.metrics),metric_path=str(a.metrics),normal_opens=0,optimizer_updates=0,selection_changes=0,official_test_access=False,seconds=time.perf_counter()-start))
print(json.dumps(s,indent=2),flush=True)
