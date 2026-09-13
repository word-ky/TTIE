"""Independent binary-head replay; replay stage never opens targets/references."""
import argparse
import hashlib
import itertools
import json
import statistics
import subprocess
from datetime import datetime,timezone
from pathlib import Path

def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def write(p,v):p.write_bytes((json.dumps(v,indent=2)+'\n').encode())
def origin(o):
    b=subprocess.check_output(['git','show',o['commit']+':'+o['path']]);assert hashlib.sha256(b).hexdigest()==o['sha256'];return json.loads(b)

def replay(out):
    import torch
    from torch import nn
    torch.set_num_threads(1);cfg=read(out/'config.json');origins=cfg['origins'];frozen=read(out/'OOF_frozen.json')
    for path,h in frozen['files_sha256'].items():assert sha(out/path)==h
    for path,h in cfg['source_code_sha256'].items():assert hashlib.sha256(subprocess.check_output(['git','show',cfg['source_sha']+':'+path])).hexdigest()==h
    # Allowlist: none of B target/table or D diagnostic artifacts are opened.
    f=torch.tensor([r['features'] for r in origin(origins['features'])]);assert f.shape==(120,5,28)
    ff=origin(origins['feature_freeze']);assert ff['features_sha256']==origins['features']['sha256']
    cf=origin(origins['C_freeze']);folds=origin(origins['C_folds']);old=origin(origins['C_decisions'])
    assert cf['decisions_sha256']==origins['C_decisions']['sha256'] and cf['folds_sha256']==origins['C_folds']['sha256']
    necessity={p['row_index']:{a:p[a+'_class']!=0 for a in ('x','y')} for p in old};del old
    assert read(out/'necessity.json')=={str(k):v for k,v in necessity.items()}
    z={'x':torch.cat([f[:,0],f[:,1]-f[:,0],f[:,2]-f[:,0]],1),'y':torch.cat([f[:,0],f[:,3]-f[:,0],f[:,4]-f[:,0]],1)}
    hard=list(itertools.product([.4,.5,.6],repeat=2));result={};heads=0
    for fold in folds:
        n=fold['fold'];train=fold['train'];held=fold['heldout'];pred={}
        fr=origin(origins[f'fold{n}_freeze']);assert origins[f'fold{n}_freeze']['sha256']==cf['fold_frozen_hashes'][str(n)]
        for a in ('x','y'):
            folder=out/f'fold{n}'/a;r=read(folder/'receipt.json')
            for path,h in r['files_sha256'].items():assert sha(folder/path)==h
            assert r['train_rows']==train and r['heldout_rows']==held and r['heldout_labels_read'] is False
            assert r['train_image_ids']==fold['train_image_ids'] and r['heldout_image_ids']==fold['heldout_image_ids']
            norm=origin(origins[f'fold{n}_{a}_norm']);assert norm==fr['heads'][a]['normalization']==read(folder/'normalization.json')
            assert origins[f'fold{n}_{a}_norm']['sha256']==fr['heads'][a]['files_sha256']['normalization.json']
            checkpoint=torch.load(folder/'head.pt',map_location='cpu',weights_only=True);state=checkpoint['state_dict'];assert checkpoint['recipe']==cfg['recipe']
            mean=z[a][train].double().mean(0).float();scale=z[a][train].double().std(0,unbiased=False).clamp_min(1e-12).float()
            assert torch.equal(mean,state['x_mean']) and torch.equal(scale,state['x_scale'])
            assert mean.tolist()==norm['mean'] and scale.tolist()==norm['scale']
            model=nn.Sequential(nn.Linear(84,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,2)).eval()
            model.load_state_dict({k.removeprefix('net.'):v for k,v in state.items() if k.startswith('net.')})
            with torch.no_grad():logits=model((z[a][held]-mean)/scale)
            classes=logits.argmax(1).tolist();values=[(.4,.6)[c] if necessity[i][a] else .5 for i,c in zip(held,classes)]
            assert read(folder/'predictions.json')==dict(row_indices=held,logits=logits.tolist(),classes=classes,values=values)
            assert sha(folder/'head.pt')==frozen['head_hashes'][f'fold{n}/{a}'];heads+=1
            assert len(read(folder/'history.json'))==100
            pred[a]=(logits.tolist(),classes,values)
        for j,i in enumerate(held):
            x,y=pred['x'][2][j],pred['y'][2][j]
            result[i]=dict(row_index=i,fold=n,x_binary_logits=pred['x'][0][j],y_binary_logits=pred['y'][0][j],x_binary_class=pred['x'][1][j],y_binary_class=pred['y'][1][j],x_move=necessity[i]['x'],y_move=necessity[i]['y'],bx=x,by=y,hard_index=hard.index((x,y)))
    assert heads==10 and [result[i] for i in range(120)]==read(out/'decisions.json')
    write(out/'head_replay.json',dict(passed=True,completed_utc=now(),heads=10,exact_predictions=120,necessity_unchanged=True,normalization_unchanged=True,heldout_references_read=False,decisions_sha256=sha(out/'decisions.json')))
    print('Independent 10-head, normalization and 120 combined prediction replay PASS')

def metrics(out):
    cfg=read(out/'config.json');o=cfg['origins'];freeze=read(out/'OOF_frozen.json');receipt=read(out/'evaluation_receipt.json')
    assert freeze['finalized_utc']<read(out/'head_replay.json')['completed_utc']<receipt['reference_opened_utc']
    assert sha(out/'decisions.json')==receipt['decisions_sha256'] and sha(out/'head_replay.json')==receipt['head_replay_sha256']
    t=origin(o['B_targets']);table=origin(o['B_table']);old=origin(o['C_decisions']);ceiling=origin(o['D_summary'])['counterfactuals']['B']
    counts=[]
    for fold in read(out/'folds.json'):
        for a in ('x','y'):
            indices=[i for i in fold['train'] if t[i]['b'+a]!=.5];labels=[int(t[i]['b'+a]==.6) for i in indices]
            assert read(out/f"fold{fold['fold']}"/a/'training_subset.json')==dict(row_indices=indices,classes=labels)
            counts.append(dict(fold=fold['fold'],axis=a,lower=labels.count(0),upper=labels.count(1),total=len(labels)))
    assert counts==read(out/'subset_counts.json') and all(r['lower'] and r['upper'] for r in counts)
    evaluated=read(out/'evaluation.json');d=read(out/'decisions.json');assert len(evaluated)==120
    hard=list(itertools.product([.4,.5,.6],repeat=2))
    for i,r in enumerate(evaluated):
        assert all(r[k]==v for k,v in d[i].items())
        assert r['row_index']==t[i]['row_index']==table[i]['row_index']==i
        assert r['image_id']==t[i]['image_id'] and r['condition']==t[i]['condition']
        assert r['H0']==table[i]['candidate_mse'][4] and r['H']==table[i]['candidate_mse'][hard.index((r['bx'],r['by']))]
        assert r['H_star']==min(table[i]['candidate_mse']) and r['C_H']==table[i]['candidate_mse'][old[i]['score_index']]
        assert r['movement']==['no_move','y_only','x_only','both'][2*int(r['bx']!=.5)+int(r['by']!=.5)]
        for a in ('x','y'):
            target=t[i]['b'+a]
            assert (r['b'+a]!=.5)==(old[i]['b'+a]!=.5)
            assert r[a+'_target_move']==(target!=.5) and r[a+'_sign_match']==((.4,.6)[r[a+'_binary_class']]==target)
            assert r[a+'_wrong']==(target!=.5 and r['b'+a]!=.5 and r['b'+a]!=target)
            assert r[a+'_C_wrong']==(target!=.5 and old[i]['b'+a]!=.5 and old[i]['b'+a]!=target)
    summary=read(out/'summary.json');mechanism=read(out/'mechanism.json');vector=[]
    for name,g in summary['groups'].items():
        rs=evaluated if name=='nonspatial_pool' else [r for r in evaluated if r['condition']==name]
        m={k:statistics.mean(r[k] for r in rs) for k in ('H0','H','H_star')};assert m==g['mse'] and len(rs)==g['count']
        assert g['ratios']==dict(H_over_H0=m['H']/m['H0'],H_over_H_star=m['H']/m['H_star'])
        assert g['outcomes']==dict(beneficial=sum(r['H']<r['H0'] for r in rs),equal=sum(r['H']==r['H0'] for r in rs),harmful=sum(r['H']>r['H0'] for r in rs))
        assert g['movements']=={k:sum(r['movement']==k for r in rs) for k in ('no_move','x_only','y_only','both')}
        expected=dict(sign_agreement={a:dict(correct=sum(r[a+'_sign_match'] for r in rs if r[a+'_target_move']),total=sum(r[a+'_target_move'] for r in rs)) for a in ('x','y')},wrong_directions={a:dict(C=sum(r[a+'_C_wrong'] for r in rs),E=sum(r[a+'_wrong'] for r in rs)) for a in ('x','y')},harmful=dict(C=sum(r['C_H']>r['H0'] for r in rs),E=g['outcomes']['harmful']),oracle_B=dict(H=ceiling['groups'][name]['mse']['H'],E_minus_B=m['H']-ceiling['groups'][name]['mse']['H'],E_over_B=m['H']/ceiling['groups'][name]['mse']['H']))
        assert expected==mechanism[name];vector.append(m['H']<=1.01*m['H0'])
    vector.append(summary['groups']['clean']['outcomes']['harmful']==0)
    assert vector==summary['acceptance_vector']==list(summary['clauses'].values()) and sum(vector)==summary['passed']
    assert summary['verdict']==('positive 5/5' if all(vector) else f'negative {sum(vector)}/5')
    write(out/'metric_verification.json',dict(passed=True,completed_utc=now(),exact_rows=120,subsets_exact=True,all_mechanism_metrics_exact=True,acceptance_vector=vector,verdict=summary['verdict']))
    print('Independent metrics and mechanism replay PASS',vector)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['replay','metrics']);p.add_argument('--output',type=Path,required=True);a=p.parse_args();globals()[a.stage](a.output)
