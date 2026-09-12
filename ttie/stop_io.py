"""Persist label-free T012 decisions separately from reference-only evaluations."""
import json
from pathlib import Path
import torch
from .stop_trajectory import checkpoint
from .stop_receipt import sha
from .restoration_metrics import evaluate_outputs
from .residual_pilot import write


def save_label_free(directory,trajectory,results,decision=None):
    directory.mkdir(parents=True)
    torch.save(trajectory['images'],directory/'checkpoint_images.pt')
    small={k:trajectory[k] for k in ('scores','grids','states','features')}
    torch.save(small,directory/'trajectory.pt')
    torch.save({n:{k:r[k].cpu().clone() for k in ('image','raw','grid')} for n,r in results.items()},directory/'outputs.pt')
    write(directory/'decisions.json',dict(gate=trajectory['gate'],trajectory=trajectory['diagnostics'],selection=decision,
          methods={n:r['diagnostics'] for n,r in results.items()}))
    receipts={name:dict(file=name,bytes=(directory/name).stat().st_size,sha256=sha(directory/name))
              for name in ('checkpoint_images.pt','trajectory.pt','outputs.pt','decisions.json')}
    write(directory/'label_free_receipt.json',receipts)
    return receipts


def evaluate_reference(directory,trajectory,results,clean,*,image_id,condition):
    rows=evaluate_outputs(results,clean,image_id=image_id,condition=condition)
    checkpoints={'identity':results['identity'],**{f'checkpoint_{i:02d}':checkpoint(trajectory,i) for i in range(len(trajectory['images']))}}
    checkpoint_rows=evaluate_outputs(checkpoints,clean,image_id=image_id,condition=condition)[1:]
    for i,row in enumerate(checkpoint_rows):row['selected_step']=i
    write(directory/'checkpoint_metrics.json',checkpoint_rows);write(directory/'metrics.json',rows)
    return rows,checkpoint_rows


def load_trajectory(directory):
    small=torch.load(directory/'trajectory.pt',map_location='cpu',weights_only=True)
    small['images']=torch.load(directory/'checkpoint_images.pt',map_location='cpu',weights_only=True)
    decisions=json.loads((directory/'decisions.json').read_text())
    small['gate']=decisions['gate'];small['diagnostics']=decisions['trajectory']
    return small


def renamed_row(row,method):return dict(row,method=method)
