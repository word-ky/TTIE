"""REFERENCE_ORACLE_ONLY: isolated T028-A validation diagnostic, never deployable TTT."""
import argparse
import csv
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from ttie.semantic_ttt import Region2
from ttie.gamma_range_box import Gamma05Box
from ttie.ssim_transfer import rgb_ssim

LABEL = 'REFERENCE_ORACLE_ONLY'
SPLIT_SHA = 'b88c8347005984b5523b117b52c0c068672fe172eb7c9aa5b60102d350e2d85b'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value): Path(path).write_text(json.dumps(value, indent=2)+'\n')
def utc(): return datetime.now(timezone.utc).isoformat()
def native(path):
    with Image.open(path) as im: a=np.asarray(im.convert('RGB')).copy()
    assert a.shape == (400,600,3)
    return torch.from_numpy(a).permute(2,0,1).unsqueeze(0).float()/255.
def mse(output, target): return (output.double()-target.double()).square().mean()
def score(output, target):
    a=output.detach().cpu().squeeze(0).permute(1,2,0).numpy()
    b=target.detach().cpu().squeeze(0).permute(1,2,0).numpy()
    value=float(np.mean((a.astype(np.float64)-b.astype(np.float64))**2))
    return dict(mse=value,psnr=-10*math.log10(value),ssim=rgb_ssim(a,b))
def frozen_model(decision, image):
    obj=SimpleNamespace(active=torch.tensor(decision['gate']['active'],device=image.device,dtype=torch.bool),
                        winner=torch.tensor(decision['gate']['winner'],device=image.device))
    model=Region2(obj.active).to(image); box=Gamma05Box(obj,2)
    saved=decision['diagnostics']['action_box']
    assert torch.equal(box.lower.cpu(),torch.tensor(saved['lower']))
    assert torch.equal(box.upper.cpu(),torch.tensor(saved['upper']))
    return model,box

def optimize_start(model, box, low, reference, initial, steps=500):
    """Exactly steps Adam updates; retain earliest lowest-MSE state over 0..steps."""
    with torch.no_grad(): model.raw.copy_(initial)
    optimizer=torch.optim.Adam([model.raw],lr=.05)
    history=[]; raw_history=[]; best=float('inf'); best_step=0; best_raw=None
    for step in range(steps+1):
        output=model(low); loss=mse(output,reference)
        value=float(loss.detach()); assert math.isfinite(value)
        history.append(value)
        grid=model.physical_grid()[:,:2]
        assert torch.isfinite(model.raw).all() and torch.isfinite(output).all()
        assert bool((grid>=box.lower-1e-6).all() and (grid<=box.upper+1e-6).all())
        raw_history.append(model.raw.detach().cpu().clone())
        if value<best:
            best=value;best_step=step;best_raw=model.raw.detach().clone()
        if step==steps: break
        optimizer.zero_grad(set_to_none=True);loss.backward()
        assert torch.isfinite(model.raw.grad).all()
        optimizer.step();box(model)
    final_raw=model.raw.detach().clone()
    with torch.no_grad(): model.raw.copy_(best_raw)
    return dict(label=LABEL,updates=steps,best_step=best_step,best_mse=best,
                initial_mse=history[0],final_mse=history[-1],history=history,raw_history=torch.stack(raw_history),
                best_raw=best_raw.cpu(),final_raw=final_raw.cpu())
