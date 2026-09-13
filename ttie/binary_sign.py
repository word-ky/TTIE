"""T020-E binary sign probe with immutable T020-C OOF necessity."""
import argparse
import json
import platform
import subprocess
from pathlib import Path
import torch
from torch import nn
from .deadband_probe import training_targets
from .direction_probe import RECIPE as DONOR_RECIPE, axis_features
from .local_geometry import now,sha,write
from .movement_attribution import summarize,HARD

RECIPE=dict(DONOR_RECIPE,output_dim=2,class_order=[.4,.6],
    normalization='reuse T020-C fold all-training-row mean/scale unchanged',
    supervision='only fold-training non-center axis targets')
INPUTS=Path('research_log/T020E_inputs.json')
SOURCES=('ttie/binary_sign.py','ttie/deadband_probe.py','ttie/direction_probe.py',
         'ttie/local_geometry.py','ttie/movement_attribution.py','research_log/T020E_inputs.json',
         'research_log/T020E_verify.py','tests/test_binary_sign.py')

def read(p):return json.loads(Path(p).read_bytes())
def raw(key,origins):
    import hashlib
    o=origins[key];b=subprocess.check_output(['git','show',o['commit']+':'+o['path']])
    assert hashlib.sha256(b).hexdigest()==o['sha256'];return b
def artifact(key,origins):return json.loads(raw(key,origins))

def axis_inputs(rows):
    f=torch.zeros(120,9,28)
    for i,r in enumerate(rows):
        assert r['row_index']==i
        f[i,[4,1,7,3,5]]=torch.tensor(r['features'])
    return axis_features(f)

def subset(target_bytes,train,axis):
    labels=training_targets(target_bytes,train)
    indices=[i for i in train if labels[i]['b'+axis]!=.5]
    return indices,[0 if labels[i]['b'+axis]==.4 else 1 for i in indices]

class BinaryHead(nn.Module):
    def __init__(self,norm):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(84,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,2))
        self.register_buffer('x_mean',torch.tensor(norm['mean'],dtype=torch.float32))
        self.register_buffer('x_scale',torch.tensor(norm['scale'],dtype=torch.float32))
    def forward(self,z):return self.net((z.to(self.x_mean)-self.x_mean)/self.x_scale)

def fit(z,indices,labels,norm):
    # Literal donor loop: only output count, training subset and fixed norm differ.
    torch.manual_seed(7);torch.set_num_threads(1)
    head=BinaryHead(norm);x=z[indices].detach().cpu().float();y=torch.tensor(labels)
    optimizer=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=1e-4)
    generator=torch.Generator().manual_seed(7);history=[]
    for epoch in range(100):
        order=torch.randperm(len(x),generator=generator);total=0.
        for batch in order.split(256):
            loss=nn.functional.cross_entropy(head(x[batch]),y[batch]);optimizer.zero_grad(set_to_none=True)
            loss.backward();optimizer.step();total+=float(loss.detach())*len(batch)
        history.append(dict(epoch=epoch+1,train_cross_entropy=total/len(x)))
    head.zero_grad(set_to_none=True);head.eval().requires_grad_(False)
    return head,history

@torch.no_grad()
def predict(head,z,necessity):
    logits=head(z);classes=logits.argmax(1).tolist()
    return logits.tolist(),classes,[.5 if not move else (.4,.6)[c] for c,move in zip(classes,necessity)]

def train(output,source):
    origins=read(INPUTS);code={}
    for path in SOURCES:
        assert Path(path).read_bytes()==subprocess.check_output(['git','show',source+':'+path]);code[path]=sha(Path(path))
    cf=artifact('C_freeze',origins);cc=artifact('C_config',origins);ff=artifact('feature_freeze',origins)
    for key,name in [('C_decisions','decisions'),('C_config','config'),('C_folds','folds')]:assert origins[key]['sha256']==cf[name+'_sha256']
    for key,name in [('features','features'),('feature_config','config')]:assert origins[key]['sha256']==ff[name+'_sha256']
    assert cc['feature_files_sha256']['features.json']==origins['features']['sha256']
    assert cc['target_origin']['sha256']==origins['B_targets']['sha256']
    br=artifact('B_receipt',origins);assert br['evaluation_sha256']==origins['B_targets']['sha256']
    z=axis_inputs(artifact('features',origins));folds=artifact('C_folds',origins)
    # Drop all donor logits/signs; retain only row-indexed necessity booleans.
    necessity={p['row_index']:{a:p[a+'_class']!=0 for a in ('x','y')} for p in artifact('C_decisions',origins)}
    target_bytes=raw('B_targets',origins);output.mkdir(parents=True)
    write(output/'config.json',dict(task='T020-E',source_sha=source,source_code_sha256=code,origins=origins,recipe=RECIPE,
        runtime=dict(torch=torch.__version__,python=platform.python_version(),device='cpu'),development_only=True))
    (output/'folds.json').write_bytes(raw('C_folds',origins));write(output/'necessity.json',necessity)
    counts=[]
    for fold in folds:
        for a in ('x','y'):
            indices,labels=subset(target_bytes,fold['train'],a)
            counts.append(dict(fold=fold['fold'],axis=a,lower=labels.count(0),upper=labels.count(1),total=len(labels)))
    write(output/'subset_counts.json',counts)
    if any(not r['lower'] or not r['upper'] for r in counts):
        write(output/'unsupported.json',dict(status='structurally unsupported',completed_utc=now(),heads_trained=0));print('structurally unsupported');return
    decisions=[];head_hashes={}
    for fold in folds:
        n=fold['fold'];fr=artifact(f'fold{n}_freeze',origins)
        assert origins[f'fold{n}_freeze']['sha256']==cf['fold_frozen_hashes'][str(n)]
        predictions={}
        for ai,a in enumerate(('x','y')):
            key=f'fold{n}_{a}_norm';norm=artifact(key,origins)
            assert origins[key]['sha256']==fr['heads'][a]['files_sha256']['normalization.json']
            assert norm==fr['heads'][a]['normalization']
            assert torch.equal(torch.tensor(norm['mean']),z[fold['train'],ai].double().mean(0).float())
            assert torch.equal(torch.tensor(norm['scale']),z[fold['train'],ai].double().std(0,unbiased=False).clamp_min(1e-12).float())
            indices,labels=subset(target_bytes,fold['train'],a)
            head,history=fit(z[:,ai],indices,labels,norm)
            folder=output/f'fold{n}'/a;folder.mkdir(parents=True)
            torch.save(dict(state_dict=head.state_dict(),recipe=RECIPE),folder/'head.pt')
            (folder/'normalization.json').write_bytes(raw(key,origins))
            write(folder/'training_subset.json',dict(row_indices=indices,classes=labels));write(folder/'history.json',history)
            logits,classes,values=predict(head,z[fold['heldout'],ai],[necessity[i][a] for i in fold['heldout']])
            predictions[a]=(logits,classes,values)
            write(folder/'predictions.json',dict(row_indices=fold['heldout'],logits=logits,classes=classes,values=values))
            hashes={p.name:sha(p) for p in folder.iterdir() if p.is_file()}
            write(folder/'receipt.json',dict(finalized_utc=now(),files_sha256=hashes,normalization_origin=origins[key],
                train_rows=fold['train'],heldout_rows=fold['heldout'],train_image_ids=fold['train_image_ids'],heldout_image_ids=fold['heldout_image_ids'],
                lower_count=labels.count(0),upper_count=labels.count(1),heldout_labels_read=False))
            head_hashes[f'fold{n}/{a}']=hashes['head.pt']
        for j,i in enumerate(fold['heldout']):
            x,y=(predictions[a][2][j] for a in ('x','y'))
            decisions.append(dict(row_index=i,fold=n,x_binary_logits=predictions['x'][0][j],y_binary_logits=predictions['y'][0][j],
                x_binary_class=predictions['x'][1][j],y_binary_class=predictions['y'][1][j],x_move=necessity[i]['x'],y_move=necessity[i]['y'],bx=x,by=y,hard_index=HARD.index((x,y))))
        print('Fold',n,'binary heads frozen',flush=True)
    decisions.sort(key=lambda r:r['row_index']);write(output/'decisions.json',decisions)
    write(output/'OOF_frozen.json',dict(finalized_utc=now(),head_hashes=head_hashes,heldout_reference_read=False,
        files_sha256={p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()}))
    print('120 OOF decisions frozen',sha(output/'decisions.json'),flush=True)

def evaluate(output):
    frozen=read(output/'OOF_frozen.json');replay=read(output/'head_replay.json');assert replay['passed']
    for path,h in frozen['files_sha256'].items():assert sha(output/path)==h
    opened=now();cfg=read(output/'config.json');origins=cfg['origins']
    targets=artifact('B_targets',origins);table=artifact('B_table',origins);old=artifact('C_decisions',origins);ceiling=artifact('D_summary',origins)['counterfactuals']['B']
    assert origins['B_table']['sha256']==artifact('B_receipt',origins)['candidate_table_sha256']
    rows=[]
    for d,t,ref,p in zip(read(output/'decisions.json'),targets,table,old):
        assert d['row_index']==t['row_index']==ref['row_index']==p['row_index']
        assert ref['candidate_mse']==t['candidate_mse'];x,y=d['bx'],d['by']
        r=dict(**d,image_id=t['image_id'],condition=t['condition'],H0=t['H0'],H=ref['candidate_mse'][d['hard_index']],H_star=t['H_star'],
            movement='no_move' if x==y==.5 else 'x_only' if y==.5 else 'y_only' if x==.5 else 'both',C_H=ref['candidate_mse'][p['score_index']])
        for a in ('x','y'):
            r[a+'_target_move']=t['b'+a]!=.5
            r[a+'_sign_match']=(.4,.6)[d[a+'_binary_class']]==t['b'+a]
            r[a+'_wrong']=t['b'+a]!=.5 and d['b'+a]!=.5 and d['b'+a]!=t['b'+a]
            r[a+'_C_wrong']=t['b'+a]!=.5 and p['b'+a]!=.5 and p['b'+a]!=t['b'+a]
            assert (d['b'+a]!=.5)==(p['b'+a]!=.5)
        rows.append(r)
    report=summarize(rows);mechanism={}
    for name,g in report['groups'].items():
        rs=rows if name=='nonspatial_pool' else [r for r in rows if r['condition']==name]
        mechanism[name]=dict(sign_agreement={a:dict(correct=sum(r[a+'_sign_match'] for r in rs if r[a+'_target_move']),total=sum(r[a+'_target_move'] for r in rs)) for a in ('x','y')},
            wrong_directions={a:dict(C=sum(r[a+'_C_wrong'] for r in rs),E=sum(r[a+'_wrong'] for r in rs)) for a in ('x','y')},
            harmful=dict(C=sum(r['C_H']>r['H0'] for r in rs),E=g['outcomes']['harmful']),
            oracle_B=dict(H=ceiling['groups'][name]['mse']['H'],E_minus_B=g['mse']['H']-ceiling['groups'][name]['mse']['H'],E_over_B=g['mse']['H']/ceiling['groups'][name]['mse']['H']))
    report['verdict']='positive 5/5' if report['all_pass'] else f"negative {report['passed']}/5"
    write(output/'evaluation.json',rows);write(output/'summary.json',report);write(output/'mechanism.json',mechanism)
    write(output/'evaluation_receipt.json',dict(reference_opened_utc=opened,completed_utc=now(),decisions_sha256=sha(output/'decisions.json'),head_replay_sha256=sha(output/'head_replay.json')))
    print(report['verdict'],report['acceptance_vector'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','evaluate']);p.add_argument('--output',type=Path,required=True);p.add_argument('--source-sha');a=p.parse_args()
    if a.stage=='train':train(a.output,a.source_sha)
    else:evaluate(a.output)
