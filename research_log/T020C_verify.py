"""Independent T020-C replay; prediction stage never opens reference targets."""
import argparse
import hashlib
import json
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n',encoding='utf8')
CLASSES=[.5,.4,.6]
HARD=[(x,y,0.) for x in (.4,.5,.6) for y in (.4,.5,.6)]


def replay(output,features):
    import torch
    from torch import nn
    torch.set_num_threads(1)
    cfg=read(output/'config.json'); freeze=read(output/'OOF_frozen.json')
    lock=read('research_log/T020C_lock.json')
    assert cfg['recipe']==lock['recipe']
    for name in ('decisions','config','folds'):assert sha(output/(name+'.json'))==freeze[name+'_sha256']
    for name,h in cfg['feature_files_sha256'].items():assert sha(features/name)==h
    fc=read(features/'config.json'); ff=read(features/'features_frozen.json')
    assert ff['features_sha256']==sha(features/'features.json') and ff['config_sha256']==sha(features/'config.json')
    assert fc['feature_schema']==lock['feature_schema'] and not fc['target_or_mse_read'] and not fc['reference_pixels_read']
    donor=read('research_log/T020C_inputs/donor_config.json')
    assert cfg['recipe']==donor['recipe']
    assert cfg['folds_origin']['sha256']==donor['input_artifact_hashes']['folds']['sha256']
    for path,h in cfg['source_code_sha256'].items():
        assert sha(path)==h==hashlib.sha256(subprocess.check_output(['git','show',cfg['source_sha']+':'+path])).hexdigest()
    for path,h in donor['source_code_sha256'].items():assert cfg['source_code_sha256'][path]==h
    assert sha(output/'folds.json')==cfg['folds_origin']['sha256']
    f=torch.tensor([r['features'] for r in read(features/'features.json')]);assert f.shape==(120,5,28)
    z={'x':torch.cat((f[:,0],f[:,1]-f[:,0],f[:,2]-f[:,0]),1),
       'y':torch.cat((f[:,0],f[:,3]-f[:,0],f[:,4]-f[:,0]),1)}
    entries=read('research_log/T020C_inputs/episodes.json'); reconstructed={};heads={}
    for fold in read(output/'folds.json'):
        train=fold['train'];test=fold['heldout'];folder=output/f"fold{fold['fold']}"
        for part in ('train','heldout'):
            assert [i for i,r in enumerate(entries) if r['image_id'] in fold[part+'_image_ids']]==fold[part]
        assert len(train)==96 and len(test)==24 and set(train).isdisjoint(test)
        fr=read(folder/'fold_frozen.json');assert sha(folder/'fold_frozen.json')==freeze['fold_frozen_hashes'][str(fold['fold'])]
        preds={}
        for a in ('x','y'):
            sub=folder/a;r=read(sub/'receipt.json');assert r==fr['heads'][a]
            for name,h in r['files_sha256'].items():assert sha(sub/name)==h
            assert r['epochs']==100 and r['heldout_target_read'] is False
            state=torch.load(sub/'head.pt',map_location='cpu',weights_only=True)
            assert state['recipe']==cfg['recipe'];s=state['state_dict']
            mean=z[a][train].double().mean(0).float();scale=z[a][train].double().std(0,unbiased=False).clamp_min(1e-12).float()
            assert torch.equal(mean,s['x_mean']) and torch.equal(scale,s['x_scale'])
            assert r['normalization']==dict(mean=mean.tolist(),scale=scale.tolist())==read(sub/'normalization.json')
            model=nn.Sequential(nn.Linear(84,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,3)).eval()
            model.load_state_dict({k.removeprefix('net.'):v for k,v in s.items() if k.startswith('net.')})
            with torch.no_grad():logits=model((z[a][test]-mean)/scale)
            preds[a]=dict(row_indices=test,logits=logits.tolist(),classes=logits.argmax(1).tolist())
            assert preds[a]==read(sub/'predictions.json')
            heads[f"fold{fold['fold']}/{a}"]=sha(sub/'head.pt')
        fd=[]
        for j,i in enumerate(test):
            cx=preds['x']['classes'][j];cy=preds['y']['classes'][j];bx=CLASSES[cx];by=CLASSES[cy];idx=HARD.index((bx,by,0.))
            d=dict(row_index=i,fold=fold['fold'],x_logits=preds['x']['logits'][j],y_logits=preds['y']['logits'][j],x_class=cx,y_class=cy,bx=bx,by=by,score_index=idx,hard_index=3*idx)
            fd.append(d);reconstructed[i]=d
        assert fd==read(folder/'decisions.json')
    assert len(heads)==10 and [reconstructed[i] for i in range(120)]==read(output/'decisions.json')
    result=dict(passed=True,completed_utc=now(),head_hashes=heads,exact_logits_classes_boundaries=120,
        normalization_training_only=True,heldout_targets_read=False,decisions_sha256=sha(output/'decisions.json'))
    write(output/'head_replay.json',result);print('Independent feature/normalizer/10-head/prediction replay PASS')


def metrics(output,features):
    cfg=read(output/'config.json'); freeze=read(output/'OOF_frozen.json');receipt=read(output/'evaluation_receipt.json')
    assert freeze['finalized_utc']<read(output/'head_replay.json')['completed_utc']<receipt['reference_opened_utc']
    origin=cfg['target_origin'];raw=subprocess.check_output(['git','show',origin['commit']+':'+origin['path']])
    assert hashlib.sha256(raw).hexdigest()==origin['sha256']==receipt['reference_sha256']
    refs=json.loads(raw);decisions=read(output/'decisions.json');rows=[]
    for fold in read(output/'folds.json'):
        assert read(output/f"fold{fold['fold']}"/'train_targets.json')=={str(i):{a:refs[i][a] for a in ('bx','by')} for i in fold['train']}
    for d,t in zip(decisions,refs):
        bx,by=d['bx'],d['by'];idx=HARD.index((bx,by,0.));xm=bx==t['bx'];ym=by==t['by']
        assert d['row_index']==t['row_index'] and idx==d['score_index']
        rows.append(dict(**d,image_id=t['image_id'],condition=t['condition'],H0=t['H0'],H1=t['candidate_mse'][idx],H_star=t['H_star'],
            target_bx=t['bx'],target_by=t['by'],x_match=xm,y_match=ym,joint_match=xm and ym,
            movement='no_move' if bx==.5 and by==.5 else 'x_only' if by==.5 else 'y_only' if bx==.5 else 'both'))
    assert rows==read(output/'evaluation.json');summary=read(output/'summary.json');vector=[]
    for name,g in summary['groups'].items():
        rs=rows if name=='nonspatial_pool' else [r for r in rows if r['condition']==name]
        m={k:statistics.mean(r[k] for r in rs) for k in ('H0','H1','H_star')}
        assert m==g['mse'] and g['count']==len(rs)
        assert g['ratios']==dict(H1_over_H0=m['H1']/m['H0'],H1_over_H_star=m['H1']/m['H_star'])
        assert g['outcomes']==dict(beneficial=sum(r['H1']<r['H0'] for r in rs),equal=sum(r['H1']==r['H0'] for r in rs),harmful=sum(r['H1']>r['H0'] for r in rs))
        assert g['movements']=={v:sum(r['movement']==v for r in rs) for v in ('no_move','x_only','y_only','both')}
        assert g['classes']=={a:{n:sum(r[a+'_class']==c for r in rs) for c,n in enumerate(('center','lower','upper'))} for a in ('x','y')}
        assert g['agreement']=={a:dict(count=sum(r[a+'_match'] for r in rs),fraction=statistics.mean(r[a+'_match'] for r in rs)) for a in ('x','y','joint')}
        vector.append(m['H1']<=1.01*m['H0'])
    vector.append(all(r['H1']<=r['H0'] for r in rows if r['condition']=='clean'))
    assert vector==summary['acceptance_vector']==list(summary['clauses'].values()) and sum(vector)==summary['passed']
    write(output/'metric_verification.json',dict(passed=True,completed_utc=now(),rows=120,all_diagnostics_exact=True,acceptance_vector=vector))
    print('Independent metrics PASS',vector)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['replay','metrics']);p.add_argument('--output',type=Path,required=True);p.add_argument('--features',type=Path,required=True);a=p.parse_args()
    globals()[a.stage](a.output,a.features)
