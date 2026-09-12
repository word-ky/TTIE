"""T013 label-free persistence, then reference-only evaluation."""
import torch
from .stop_receipt import sha
from .stop_trajectory import capture,checkpoint
from .stop_pilot import controls
from .stop_io import save_label_free,renamed_row
from .restoration_metrics import evaluate_outputs
from .residual_pilot import write
from .energy_ttt import METHODS,trajectory
from .energy_metrics import PRIMARY_METHOD,ORACLE,alignment


def hashes(directory,names):
    return {n:dict(file=n,bytes=(directory/n).stat().st_size,sha256=sha(directory/n)) for n in names}


def save_bank(directory,bank):
    directory.mkdir(parents=True)
    torch.save(bank['images'],directory/'bank_images.pt')
    torch.save({k:bank[k] for k in ('states','grids','scores','features')},directory/'bank.pt')
    write(directory/'bank_decisions.json',{k:bank[k] for k in ('names','gate','semantic_diagnostics')})
    receipt=hashes(directory,('bank_images.pt','bank.pt','bank_decisions.json'));write(directory/'label_free_receipt.json',receipt)
    return receipt


def bank_targets(directory,bank,clean):
    targets=[dict(index=i,name=bank['names'][i],mse=float((pixels-clean.cpu()).square().mean())) for i,pixels in enumerate(bank['images'])]
    write(directory/'targets.json',targets);return targets


def run_label_free(image,scorer,receipt,head,*,max_steps=40):
    old=capture(image,scorer,receipt,max_steps=max_steps)
    results=controls(image,scorer,receipt,old,full=False);results.pop('region2_ttt_projected_1step')
    results['fixed_step_source']=checkpoint(old,16)
    trajectories={};decisions={}
    for method in METHODS:
        result,t,decision=trajectory(image,scorer,receipt,head,basis=method.split('_')[0],max_steps=max_steps)
        results[method]=result;trajectories[method]=t;decisions[method]=decision
    return results,old,trajectories,decisions


def save_episode(directory,results,old,trajectories,decisions):
    directory.mkdir(parents=True)
    # Every control, energy trajectory and label-free decision exists and is
    # hashed before any caller attaches clean-reference metrics/gradients.
    receipts={'semantic':save_label_free(directory/'semantic',old,{k:results[k] for k in
        ('identity','region2_direct','region2_discrete_projected','region2_ttt_projected','fixed_step_source')})}
    for method,t in trajectories.items():receipts[method]=save_label_free(directory/method,t,{method:results[method]},decisions[method])
    torch.save({m:{k:r[k].cpu() for k in ('image','raw','grid')} for m,r in results.items()},directory/'outputs.pt')
    write(directory/'decisions.json',dict(methods={m:r['diagnostics'] for m,r in results.items()},selections=decisions))
    receipts['episode']=hashes(directory,('outputs.pt','decisions.json'));write(directory/'label_free_receipt.json',receipts)
    return receipts


def evaluate_episode(directory,results,trajectories,clean,image,*,image_id,condition,diagnose):
    rows=evaluate_outputs(results,clean,image_id=image_id,condition=condition)
    t=trajectories[PRIMARY_METHOD]
    candidates={'identity':results['identity'],**{f'checkpoint_{i:02d}':checkpoint(t,i) for i in range(len(t['images']))}}
    per_step=evaluate_outputs(candidates,clean,image_id=image_id,condition=condition)[1:]
    for i,r in enumerate(per_step):r['selected_step']=i
    best=min(range(len(per_step)),key=lambda i:(per_step[i]['mse'],i))
    rows.append(renamed_row(per_step[best],ORACLE));write(directory/'checkpoint_metrics.json',per_step)
    write(directory/'oracle_diagnostic.json',dict(selected_step=best,reference_only=True,checkpoint_file=PRIMARY_METHOD+'/checkpoint_images.pt'))
    for r in rows:
        if r['method'] in trajectories:r['selected_step']=results[r['method']]['diagnostics']['selected_step']
    write(directory/'metrics.json',rows)
    diagnostic=None
    if diagnose and condition!='clean' and any(t['gate']['active']):
        diagnostic=dict(image_id=image_id,condition=condition,**alignment(image,clean,t))
        write(directory/'alignment.json',diagnostic)
    return rows,diagnostic,checkpoint(t,best)
