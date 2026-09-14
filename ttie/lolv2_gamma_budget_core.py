"""T026-B single dark-EV-bound variant; low-only native-resolution execution."""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .clip_signal import FrozenCLIP
from .learned_prototypes import Prototypes
from .semantic_ttt import SemanticScorer
from .energy_model import load_energy
from .gamma_range_ttt import trajectory
from .stop_receipt import sha


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def native_rgb(path):
    with Image.open(path) as image:
        array = np.asarray(image.convert('RGB')).copy()
    return torch.from_numpy(array).permute(2,0,1).unsqueeze(0).float()/255.


def low_image_opener(allowed,opened):
    original_open=Image.open
    def guarded(path,*args,**kwargs):
        if str(Path(path).resolve()) not in allowed:
            raise PermissionError('T026-B accepts only frozen low-light images')
        opened.append(str(Path(path).resolve()))
        return original_open(path,*args,**kwargs)
    return guarded


def save_episode(directory,result,t,decision):
    directory.mkdir()
    # Clone selected views: identical values, without the T022-A backing-storage overhead.
    torch.save({k:result[k].clone() for k in ['image','raw','grid']},directory/'output.pt')
    write(directory/'decision.json',dict(selection=decision,gate=t['gate'],diagnostics=t['diagnostics']))
    torch.save({k:t[k] for k in ['states','grids','scores','features']},directory/'trajectory.pt')
    return {name:dict(sha256=sha(directory/name),bytes=(directory/name).stat().st_size)
            for name in ['output.pt','decision.json','trajectory.pt']}


def main():
    p = argparse.ArgumentParser()
    for key in ['low-root', 'split', 'assets', 'out']:
        p.add_argument('--'+key, type=Path, required=True)
    args = p.parse_args()
    torch.manual_seed(7); torch.set_num_threads(1)
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    device='cuda:0'
    split=json.loads(args.split.read_bytes()); assets=json.loads(args.assets.read_bytes())
    assert len(split['selected'])==100
    for value in assets['files'].values():
        assert sha(Path(value['path']))==value['sha256']
    for row in split['selected']:
        assert sha(args.low_root/row['low'])==row['low_sha256']
    args.out.mkdir(parents=True, exist_ok=True)
    # Record every decoded image. This process receives low paths only; paired
    # normal images are deployed to this task evaluation directory only after completion.
    opened=[]
    allowed={str((args.low_root/r['low']).resolve()) for r in split['selected']}
    Image.open=low_image_opener(allowed,opened)
    files=assets['files']
    receipt=json.loads(Path(files['gate']['path']).read_bytes())
    encoder=FrozenCLIP.from_checkpoint(files['clip']['path'],device)
    saved=torch.load(files['prototypes']['path'],map_location=device,weights_only=True)
    scorer=SemanticScorer(encoder,Prototypes(saved['raw']))
    head=load_energy(files['energy']['path'])
    config=dict(task='T026-B',method='T014 Region2 dark EV upper2 active gamma lower0.5',dark_ev_upper=2.,active_gamma_lower=.5,assets=assets,
                split_sha256=sha(args.split),seed=7,tf32=False,max_steps=80,
                native_resolution=True,optimizer='unchanged Adam lr=.03',
                checkpoint='minimum predicted energy, earliest tie',
                gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,
                frozen_before_inference_utc=datetime.now(timezone.utc).isoformat())
    write(args.out/'config.json',config)
    rows=[]
    for i,row in enumerate(split['selected']):
        image=native_rgb(args.low_root/row['low']).to(device)
        torch.cuda.synchronize(); start=time.perf_counter()
        result,t,decision=trajectory(image,scorer,receipt,head,basis='region2',max_steps=80)
        torch.cuda.synchronize(); seconds=time.perf_counter()-start
        assert torch.isfinite(result['image']).all()
        directory=args.out/f'{i:03d}'
        files_saved=save_episode(directory,result,t,decision)
        rows.append(dict(index=i,low=row['low'],seconds=seconds,shape=list(image.shape),
                         active=bool(any(t['gate']['active'])),updates=t['diagnostics']['steps'],
                         selected_step=decision['selected_step'],files=files_saved,
                         persisted_utc=datetime.now(timezone.utc).isoformat()))
        write(args.out/'progress.json',rows)
        print(f'{i+1}/100 seconds={seconds:.3f} updates={rows[-1]["updates"]}',flush=True)
        del result,t,image
    assert opened==[str((args.low_root/r['low']).resolve()) for r in split['selected']]
    assert all(sha(Path(v['path']))==v['sha256'] for v in assets['files'].values())
    write(args.out/'freeze.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),
          rows=rows,opened_images=opened,task_references_deployed=False,low_only_decoder_enforced=True,
          split_sha256=sha(args.split),config_sha256=sha(args.out/'config.json')))
    print('All 100 outputs frozen before reference deployment',flush=True)


if __name__=='__main__':
    main()
