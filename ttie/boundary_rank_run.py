"""T016-D: ten fixed CPU ranking heads with the accepted T016-C OOF contract."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import torch
from .boundary_probe import grouped_folds,probe_features,predict
from .boundary_probe_metrics import evaluate
from .boundary_probe_run import load_inputs,sha,write,now,EVIDENCE_COMMIT,SOURCES as DONOR_SOURCES
from .boundary_rank import train_rank,RECIPE
from .routing.provenance import verify_source

C_COMMIT='433683eccad24dc763450be6a546072a72e0910b'
C_INPUTS={name:'research_log/T016C_run/'+name+'.json' for name in ('folds','config','evaluation','summary','OOF_frozen')}
SOURCES=DONOR_SOURCES+('ttie/boundary_rank.py','ttie/boundary_rank_run.py')


def load_source():
    data,hashes=load_inputs();old={};combined={f'T016B/{k}':dict(v,commit=EVIDENCE_COMMIT) for k,v in hashes.items()}
    for key,path in C_INPUTS.items():
        blob=subprocess.check_output(['git','show',C_COMMIT+':'+path]);old[key]=json.loads(blob)
        combined['T016C/'+key]=dict(commit=C_COMMIT,path=path,sha256=hashlib.sha256(blob).hexdigest())
    assert old['config']['input_artifact_hashes']==hashes
    return data,old,combined


def fit_fold(x,reference,episodes,fold,output):
    target=torch.tensor([reference[episodes[i]]['reference_mse'] for i in fold['train']],dtype=torch.float64)
    head,history,pair_info=train_rank(x[fold['train']],target)
    values,choices=predict(head,x[fold['heldout']])
    predictions=[dict(episode=episodes[i],fold=fold['fold'],predictions=v,selected_index=c)
                 for i,v,c in zip(fold['heldout'],values,choices)]
    output.mkdir(parents=True)
    torch.save(dict(state_dict=head.state_dict(),dimension=x.shape[-1],recipe=dict(RECIPE,input_dim=x.shape[-1])),output/'head.pt')
    write(output/'history.json',history);write(output/'predictions.json',predictions)
    receipt=dict(finalized_utc=now(),predictions_sha256=sha(output/'predictions.json'),head_sha256=sha(output/'head.pt'),
        history_sha256=sha(output/'history.json'),train_image_ids=fold['train_image_ids'],heldout_image_ids=fold['heldout_image_ids'],
        train_candidate_rows=len(fold['train'])*9,heldout_rows=len(predictions)*9,normalization=head.normalization(),
        final_train_pairwise_loss=history[-1]['train_pairwise_loss'],epochs=len(history),dimension=x.shape[-1],
        heldout_reference_used=False,**pair_info)
    write(output/'receipt.json',receipt);return predictions,receipt


def interpret(a,b):
    if a:return '28d_development_rankable_under_pairwise_supervision'
    if b:return 'geometry_and_pairwise_supervision_jointly_restore_development_rankability'
    return 'neither_rank_probe_establishes_safe_development_ranking'


def finish(output,predictions,training,reference,pointwise):
    frozen={}
    for name,rows in predictions.items():
        write(output/name/'oof.json',rows)
        frozen[name]=dict(oof_sha256=sha(output/name/'oof.json'),fold_receipts=training[name])
    write(output/'OOF_frozen.json',dict(finalized_utc=now(),probes=frozen,heldout_reference_evaluation=False))
    evaluation_started=now();reports={};evaluated={}
    for name,rows in predictions.items():
        assert sha(output/name/'oof.json')==frozen[name]['oof_sha256']
        report,attached=evaluate(rows,reference)
        old={r['episode']:r for r in pointwise[name.replace('rank','probe')]}
        for row in attached:row['pointwise_t016c_mse']=old[row['episode']]['selected_mse']
        for condition,g in report['groups'].items():
            group=attached if condition=='spatial_pool' else [r for r in attached if r['condition']==condition]
            g['pointwise_t016c_mse']=statistics.mean(r['pointwise_t016c_mse'] for r in group)
            g['ratios']['selected_over_pointwise_t016c']=g['selected_mse']/g['pointwise_t016c_mse']
        report['final_train_pairwise_loss']=[r['final_train_pairwise_loss'] for r in training[name]]
        report['training_pairs']=[r['non_tied_pairs'] for r in training[name]]
        reports[name]=report;evaluated[name]=attached
    comparison=dict(rank30_over_rank28=reports['rank30']['groups']['spatial_pool']['selected_mse']/reports['rank28']['groups']['spatial_pool']['selected_mse'],
        rank28_over_probe28=reports['rank28']['groups']['spatial_pool']['ratios']['selected_over_pointwise_t016c'],
        rank30_over_probe30=reports['rank30']['groups']['spatial_pool']['ratios']['selected_over_pointwise_t016c'])
    result=dict(probes=reports,comparison=comparison,interpretation=interpret(reports['rank28']['qualified'],reports['rank30']['qualified']))
    write(output/'evaluation.json',evaluated);write(output/'summary.json',result)
    write(output/'evaluation_receipt.json',dict(evaluation_started_utc=evaluation_started,
        oof_frozen_sha256=sha(output/'OOF_frozen.json'),unchanged_oof_hashes={n:sha(output/n/'oof.json') for n in predictions}))
    return result


def markdown(report):
    lines=['# T016-D: grouped OOF pairwise boundary-ranking probe','',f"Result: **{report['interpretation']}**.",'',
        'Development-only loss-alignment diagnostic. Same 40 IDs, five image-grouped folds, nine candidates and 28/30 features as T016-C. Ten fixed scalar heads; all non-tied within-episode pairs once per epoch; unweighted logistic loss, no target standardization. No fresh qualification or deployable model.','']
    for name,p in report['probes'].items():
        lines += ['## '+name,'',f"Five clauses: {p['passed']}/5. {p['clauses']}",'',
            '| Group | Selected MSE | Region2 | Hard oracle | Frozen T016-B | T016-C counterpart | /Region2 | /Hard oracle | /Frozen B | /Pointwise C |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for condition,g in p['groups'].items():
            values=[g[k] for k in ('selected_mse','region2_mse','hard_oracle_mse','frozen_t016b_mse','pointwise_t016c_mse')]+list(g['ratios'].values())
            lines.append('| '+condition+' | '+' | '.join('null' if v is None else f'{v:.12g}' for v in values)+' |')
        lines += ['','| Group | Selected counts | Oracle counts | Disagreement | Outside oracle ties | Spearman |','|---|---|---|---:|---:|---|']
        for condition,g in p['groups'].items():
            lines.append(f"| {condition} | {g['selected_counts']} | {g['oracle_counts']} | {g['disagreement_rate']} | {g['outside_oracle_tie_set_rate']} | {g['spearman']} |")
        lines += ['','| Fold | Held-out IDs | Selected MSE | LR / Quadrants / Offset | Non-tied train pairs | Final train pair loss |','|---|---|---:|---|---:|---:|']
        for f in p['folds']:
            i=f['fold'];lines.append(f"| {i} | {f['image_ids']} | {f['selected_mse']:.12g} | {f['per_condition']} | {p['training_pairs'][i]} | {p['final_train_pairwise_loss'][i]:.12g} |")
        lines += ['', 'Prediction/oracle minimum tie episodes: '+str({c:(g['prediction_tied_episodes'],g['oracle_tied_episodes']) for c,g in p['groups'].items()})+'.','']
    lines += ['## Comparisons','',str(report['comparison']),'',
        'Exact score/oracle ties choose first lexicographic boundary. Spearman uses average ranks; constant cases are null. All OOF outputs were frozen before reference evaluation. No alternate threshold, weighting, abstention, epoch, fold or model selection.','',
        'Stop and report the literal result. No larger model, spatial/image features, continuous boundaries or fresh experiment is authorized by this diagnostic.','']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-sha',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    code=verify_source(a.source_sha,SOURCES);torch.set_num_threads(1)
    data,old,hashes=load_source();entries=data['selection']['episodes'];episodes=[r['episode'] for r in entries]
    reference={r['episode']:r for r in data['reference']}
    metadata=[{k:reference[e][k] for k in ('image_id','condition')} for e in episodes]
    folds=grouped_folds(metadata);assert folds==old['folds']
    assert len(entries)==120 and all(len(f['train_image_ids'])==32 and len(f['heldout_image_ids'])==8 for f in folds)
    candidates=data['selection']['candidates'];assert candidates==[[x,y,0.] for x in (.4,.5,.6) for y in (.4,.5,.6)]
    a.output.mkdir(parents=True)
    write(a.output/'config.json',dict(task='T016-D',source_sha=a.source_sha,source_code_sha256=code,input_artifact_hashes=hashes,
        candidates=candidates,recipes={f'rank{d}':dict(RECIPE,input_dim=d) for d in (28,30)},
        input_standardization_population='all 864 training candidate rows, including episodes whose reference candidates tie',
        cpu_only=True,runtime=dict(python=platform.python_version(),torch=torch.__version__,device='cpu'),
        development_only=True,new_image_ids=0,rendering_calls=0,clip_calls=0))
    write(a.output/'folds.json',folds);predictions={};training={}
    for dimension in (28,30):
        name=f'rank{dimension}';x=probe_features([r['features'] for r in entries],candidates,dimension)
        rows=[];receipts=[]
        for fold in folds:
            result,receipt=fit_fold(x,reference,episodes,fold,a.output/name/f"fold{fold['fold']}")
            rows.extend(result);receipts.append(receipt)
            print(name,'fold',fold['fold'],'finalized; pairs',receipt['non_tied_pairs'],'loss',receipt['final_train_pairwise_loss'],flush=True)
        rows.sort(key=lambda r:episodes.index(r['episode']));assert len(rows)==len({r['episode'] for r in rows})==120
        predictions[name]=rows;training[name]=receipts
    result=finish(a.output,predictions,training,data['reference'],old['evaluation'])
    (a.output/'T016D_analysis.md').write_text(markdown(result),encoding='utf-8')
    print('T016-D complete',result['interpretation'],flush=True)


if __name__=='__main__':main()
