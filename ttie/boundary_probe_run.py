"""Fixed ten-head CPU OOF diagnostic, consuming only committed compact evidence."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import torch
from .boundary_probe import grouped_folds,probe_features,train_probe,predict
from .boundary_probe_metrics import evaluate,interpretation,report_markdown
from .energy_model import RECIPE,SCHEMA
from .routing.provenance import T015_SOURCES,verify_source

EVIDENCE_COMMIT='4062e01cb93de731c394015c5ac741d6c08e04d8'
BASE='research_log/remote_runs/20260913-013748-ttie-t016b-assets-ready/artifacts/'
INPUTS={k:BASE+v for k,v in dict(selection='scoring/selection.json',reference='evaluation/evaluation.json',
    selection_receipt='scoring/selection_receipt.json',config='scoring/config.json',
    evaluation_receipt='evaluation/evaluation_receipt.json').items()}
SOURCES=T015_SOURCES+('ttie/boundary_probe.py','ttie/boundary_probe_metrics.py','ttie/boundary_probe_run.py')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def now():return datetime.now(timezone.utc).isoformat()


def load_inputs():
    data={};hashes={}
    for key,path in INPUTS.items():
        blob=subprocess.check_output(['git','show',EVIDENCE_COMMIT+':'+path])
        hashes[key]=dict(path=path,sha256=hashlib.sha256(blob).hexdigest())
        data[key]=json.loads(blob)
    assert hashes['selection']['sha256']==data['selection_receipt']['selection_sha256']
    assert hashes['config']['sha256']==data['selection_receipt']['config_sha256']
    assert data['evaluation_receipt']['selection_receipt']==data['selection_receipt']
    return data,hashes


def train_fold(x,reference,episodes,fold,output):
    """Read training targets only. Held-out inference receives x tensors only."""
    targets=torch.tensor([reference[episodes[i]]['reference_mse'] for i in fold['train']],dtype=torch.float64).flatten()
    train_x=x[fold['train']].reshape(-1,x.shape[-1])
    head,history=train_probe(train_x,targets)
    values,selected=predict(head,x[fold['heldout']])
    predictions=[dict(episode=episodes[i],fold=fold['fold'],predictions=v,selected_index=c)
                 for i,v,c in zip(fold['heldout'],values,selected)]
    output.mkdir(parents=True)
    torch.save(dict(state_dict=head.state_dict(),dimension=x.shape[-1],recipe=dict(RECIPE,input_dim=x.shape[-1])),output/'head.pt')
    write(output/'history.json',history);write(output/'predictions.json',predictions)
    receipt=dict(finalized_utc=now(),predictions_sha256=sha(output/'predictions.json'),
        head_sha256=sha(output/'head.pt'),history_sha256=sha(output/'history.json'),
        train_image_ids=fold['train_image_ids'],heldout_image_ids=fold['heldout_image_ids'],
        train_rows=len(targets),heldout_rows=len(predictions)*9,normalization=head.normalization(),
        final_train_huber=history[-1]['train_huber'],epochs=len(history),dimension=x.shape[-1],
        heldout_reference_used=False)
    write(output/'receipt.json',receipt)
    return predictions,receipt


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-sha',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    code=verify_source(a.source_sha,SOURCES)
    torch.set_num_threads(1)
    data,hashes=load_inputs(); entries=data['selection']['episodes']
    reference={r['episode']:r for r in data['reference']}
    episodes=[r['episode'] for r in entries]
    metadata=[{k:reference[e][k] for k in ('image_id','condition')} for e in episodes]
    folds=grouped_folds(metadata)
    assert len(entries)==120 and len({r['image_id'] for r in metadata})==40
    assert all(len(f['train_image_ids'])==32 and len(f['heldout_image_ids'])==8 for f in folds)
    candidates=data['selection']['candidates']
    assert candidates==[[x,y,0.] for x in (.4,.5,.6) for y in (.4,.5,.6)]
    a.output.mkdir(parents=True)
    write(a.output/'config.json',dict(task='T016-C',source_sha=a.source_sha,source_code_sha256=code,
        evidence_commit=EVIDENCE_COMMIT,input_artifact_hashes=hashes,base_feature_schema=SCHEMA,
        candidates=candidates,recipes={f'probe{d}':dict(RECIPE,input_dim=d) for d in (28,30)},
        fold_rule='sorted unique image IDs, j modulo 5; all conditions/candidates grouped',
        training_row_order='original saved episode order, then lexicographic candidates',
        cpu_only=True,runtime=dict(python=platform.python_version(),torch=torch.__version__,device='cpu'),
        development_only=True,new_image_ids=0,rendering_calls=0,clip_calls=0))
    write(a.output/'folds.json',folds)
    all_predictions={};training={};frozen={}
    for dimension in (28,30):
        name=f'probe{dimension}';x=probe_features([r['features'] for r in entries],candidates,dimension)
        predictions=[];receipts=[]
        for fold in folds:
            out=a.output/name/f"fold{fold['fold']}"
            heldout,receipt=train_fold(x,reference,episodes,fold,out)
            predictions.extend(heldout);receipts.append(receipt)
            print(name,'fold',fold['fold'],'predictions finalized; final train Huber',receipt['final_train_huber'],flush=True)
        predictions.sort(key=lambda r:episodes.index(r['episode']))
        assert len(predictions)==len({r['episode'] for r in predictions})==120
        write(a.output/name/'oof.json',predictions)
        all_predictions[name]=predictions;training[name]=receipts
        frozen[name]=dict(oof_sha256=sha(a.output/name/'oof.json'),fold_receipts=receipts)
    write(a.output/'OOF_frozen.json',dict(finalized_utc=now(),probes=frozen,heldout_reference_evaluation=False))
    # All ten fold predictions and both OOF tables now exist. No evaluation guided fitting or selection.
    evaluation_started=now();reports={};evaluated={}
    for name,rows in all_predictions.items():
        assert sha(a.output/name/'oof.json')==frozen[name]['oof_sha256']
        report,attached=evaluate(rows,data['reference'])
        report['final_train_huber']=[r['final_train_huber'] for r in training[name]]
        reports[name]=report;evaluated[name]=attached
    median28=reports['probe28']['groups']['spatial_pool']['spearman']['median']
    median30=reports['probe30']['groups']['spatial_pool']['spearman']['median']
    comparison=dict(probe30_over_probe28=reports['probe30']['groups']['spatial_pool']['selected_mse']/reports['probe28']['groups']['spatial_pool']['selected_mse'],
        median_spearman_difference=median30-median28 if median28 is not None and median30 is not None else None)
    report=dict(probes=reports,comparison=comparison,
        interpretation=interpretation(reports['probe28']['qualified'],reports['probe30']['qualified']))
    write(a.output/'evaluation.json',evaluated);write(a.output/'summary.json',report)
    write(a.output/'evaluation_receipt.json',dict(evaluation_started_utc=evaluation_started,
        oof_frozen_sha256=sha(a.output/'OOF_frozen.json'),source_sha=a.source_sha,
        unchanged_oof_hashes={n:sha(a.output/n/'oof.json') for n in all_predictions}))
    (a.output/'T016C_analysis.md').write_text(report_markdown(report),encoding='utf-8')
    print('T016-C complete',report['interpretation'],flush=True)


if __name__=='__main__':main()
