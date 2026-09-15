"""Independent math/tie-pair/rank replay; imports no main T044 helper."""
import argparse,json,hashlib,math,csv,statistics,time
from pathlib import Path
import torch
p=argparse.ArgumentParser()
for k in ['stage-a','stage-b','accepted','metrics']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();torch.set_num_threads(1);start=time.perf_counter()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def thash(t):return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()
f=json.loads((a.stage_a/'freeze.json').read_bytes());r=json.loads((a.stage_b/'receipt.json').read_bytes())
assert sha(a.stage_a/'freeze.json')==r['stage_a_freeze_sha256'] and f['completed_utc']<r['metric_open_utc']
assert sha(a.stage_a/'scores.json')==f['scores_sha256'] and sha(a.stage_a/'states.pt')==f['states_sha256']
assert sha(a.accepted/'freeze.json')==f['prior_freeze_sha256']
rows=json.loads((a.stage_a/'scores.json').read_bytes());snap=torch.load(a.stage_a/'states.pt',map_location='cpu',weights_only=True)
scores=[];errors=[]
for row,s in zip(rows,snap):
    root=a.accepted/f'{row["index"]:03d}'/'common'
    assert sha(root/'trajectory.pt')==row['trajectory_sha256'] and sha(root/'decision.json')==row['decision_sha256']
    d=json.loads((root/'decision.json').read_bytes());t=torch.load(root/'trajectory.pt',weights_only=True,map_location='cpu')
    assert d['selection']['selected_step']==row['selected_step'] and d['gate']==s['gate']
    assert hashlib.sha256(json.dumps(d['gate'],sort_keys=True,separators=(',',':')).encode()).hexdigest()==row['gate_sha256']
    assert torch.equal(t['states'][10],s['anchor']) and torch.equal(t['states'][row['selected_step']],s['selected'])
    assert thash(s['anchor'])==row['anchor_sha256'] and thash(s['selected'])==row['selected_sha256']
    values=[]
    for j,active in enumerate(d['gate']['active']):
        if not active:continue
        y,x=divmod(j,2)
        u=[float(s[k][0,0,y,x]) for k in ['anchor','selected']]
        v=[float(s[k][0,1,y,x]) for k in ['anchor','selected']]
        ev=[2*math.tanh(z) for z in u];gamma=[math.exp(math.log(2)*math.tanh(z)) for z in v]
        values.extend([((ev[1]-ev[0])/4)**2,(math.log2(gamma[1]/gamma[0])/2)**2])
    value=math.sqrt(math.fsum(values)/len(values)) if values else 0.
    scores.append(value);errors.append(abs(value-row['D_legacy']))
assert len(scores)==100
# Prior quality data is read only after the scalar reconstruction above.
assert sha(a.metrics)==r['metric_sha256']=='cdbd7fec76194153db645133d5d35dd43f6f9d852ebbd6674756d77d1ad7aee0'
metrics=list(csv.DictReader(a.metrics.open()));deltas=[]
for row,m in zip(rows,metrics):
    assert row['low']==m['low'] and row['index']==int(m['index'])
    deltas.append(float(m['common_psnr'])-float(m['baseline_psnr']))
loss=[i for i,y in enumerate(deltas) if y<0];other=[i for i,y in enumerate(deltas) if y>=0];assert len(loss)==29 and len(other)==71
auc=math.fsum(1. if scores[i]>scores[j] else .5 if scores[i]==scores[j] else 0. for i in loss for j in other)/(29*71)
def ranks(values):
    return [1+sum(v<x for v in values)+(sum(v==x for v in values)-1)/2 for x in values]
x,y=ranks(scores),ranks(deltas);mx,my=statistics.mean(x),statistics.mean(y)
rho=math.fsum((u-mx)*(v-my) for u,v in zip(x,y))/math.sqrt(math.fsum((u-mx)**2 for u in x)*math.fsum((v-my)**2 for v in y))
classification='legacy-extrapolation risk association supported' if auc>=.75 and rho<=-.35 else 'legacy-extrapolation risk association not supported / mixed'
summary=json.loads((a.stage_b/'summary.json').read_bytes());errors.extend([abs(auc-summary['roc_auc']),abs(rho-summary['spearman'])]);assert summary['classification']==classification
def quantile(v,q):
    v=sorted(v);position=(len(v)-1)*q;lo=int(position);hi=math.ceil(position);return v[lo]+(v[hi]-v[lo])*(position-lo)
for name,indices in [('loss',loss),('non_loss',other)]:
    v=[scores[i] for i in indices]
    for k,value in dict(mean=statistics.mean(v),median=statistics.median(v),q25=quantile(v,.25),q75=quantile(v,.75)).items():errors.append(abs(value-summary['groups'][name][k]))
pairs=json.loads((a.stage_b/'pairs.json').read_bytes())
for i,row in enumerate(pairs):
    assert row['loss']==(deltas[i]<0);errors.append(abs(row['delta_psnr']-deltas[i]))
for key,indices in [('smallest',sorted(range(100),key=lambda i:(scores[i],i))[:5]),('largest',sorted(range(100),key=lambda i:(-scores[i],i))[:5])]:assert [r['index'] for r in summary[key]]==indices
assert max(errors)<=1e-10
out=dict(status='PASS',scores=100,scalar_checks=len(errors),auc_pairs=29*71,max_abs_error=max(errors),roc_auc=auc,spearman=rho,classification=classification,all_frozen_states_and_gates_exact=True,score_freeze_before_metrics=True,seconds=time.perf_counter()-start)
(a.stage_b/'independent_replay.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
