"""T023-A: one source-only Sobolev head from frozen accepted EV2 trajectories."""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import torch
from .lolv2_ev_core import native_rgb, low_image_opener, save_episode, write
from .clip_signal import FrozenCLIP
from .learned_prototypes import Prototypes
from .semantic_ttt import SemanticScorer
from .ev_range_ttt import trajectory
from .energy_model import EnergyHead, load_energy, save_energy, RECIPE
from .sobolev_source import source_derivatives, JACOBIAN
from .sobolev_train import raw_gradient, cosine, training_statistics, DERIVATIVE_LOSS
from .stop_quality import train_head
from .stop_receipt import sha


def train_primary(x, mse, records):
    jac=records['jacobian'].cpu().float()
    truth=records['reference_gradient'].cpu().float()
    mask=records['direction_mask'].cpu()
    def extra(head,prediction,features,indices):
        grad=raw_gradient(head,prediction,features,jac[indices],create_graph=True)
        valid=mask[indices]
        return ((1-cosine(grad[valid],truth[indices][valid]))/2).mean() if valid.any() else grad.sum()*0
    return train_head(x,mse,head_factory=EnergyHead,extra_loss=extra)


def main():
    parser=argparse.ArgumentParser()
    for key in ['source-root','manifest','assets','out']:
        parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(7);torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device='cuda:0';manifest=json.loads(a.manifest.read_bytes());assets=json.loads(a.assets.read_bytes())
    assert len(manifest['selected'])==16 and manifest['disjoint']
    for item in assets['files'].values():assert sha(Path(item['path']))==item['sha256']
    files=assets['files'];receipt=json.loads(Path(files['gate']['path']).read_bytes())
    scorer=SemanticScorer(FrozenCLIP.from_checkpoint(files['clip']['path'],device),
                         Prototypes(torch.load(files['prototypes']['path'],map_location=device,weights_only=True)['raw']))
    old=load_energy(files['energy']['path'])
    entries=[];xs=[];ys=[];all_records=[];opened=[]
    lows={str((a.source_root/r['low']).resolve()) for r in manifest['selected']}
    write(a.out/'config.json',dict(task='T023-A',manifest_sha256=sha(a.manifest),assets=assets,
          recipe=RECIPE,derivative_loss=DERIVATIVE_LOSS,source_trajectory='accepted C,40 steps',
          derivative='unchanged source_derivatives replays accepted EV2 raw states; no projection in derivative path',
          gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda))
    for i,row in enumerate(manifest['selected']):
        start=time.monotonic();low=a.source_root/row['low'];normal=a.source_root/row['normal']
        assert sha(low)==row['low_sha256'] and sha(normal)==row['normal_sha256']
        with patch('PIL.Image.open',low_image_opener(lows,opened)):
            image=native_rgb(low).to(device)
            result,bank,decision=trajectory(image,scorer,receipt,old,max_steps=40)
        assert len(bank['states'])==41
        folder=a.out/f'{i:03d}';frozen=save_episode(folder,result,bank,decision)
        frozen_utc=datetime.now(timezone.utc).isoformat()
        write(folder/'pre_reference_freeze.json',dict(completed_utc=frozen_utc,files=frozen,low=row['low'],states=41))
        normal_started=datetime.now(timezone.utc).isoformat()
        with patch('PIL.Image.open',low_image_opener({str(normal.resolve())},opened)):
            clean=native_rgb(normal)
        # Same float32 image-MSE targets and double log transform as T014.
        mse=torch.tensor([float((pixels-clean).square().mean()) for pixels in bank['images']],dtype=torch.float64)
        records=source_derivatives(image,clean,scorer,receipt,bank)
        assert all(torch.isfinite(v).all() for v in records.values())
        for name,r in frozen.items():assert sha(folder/name)==r['sha256']
        torch.save(dict(mse=mse,**records),folder/'source_supervision.pt')
        entries.append(dict(index=i,low=row['low'],normal=row['normal'],frozen_utc=frozen_utc,
            normal_open_started_utc=normal_started,files=frozen,supervision_sha256=sha(folder/'source_supervision.pt'),
            states=41,direction_rows=int(records['direction_mask'].sum()),seconds=time.monotonic()-start))
        write(a.out/'source_receipts.json',entries)
        xs.append(bank['features']);ys.append(mse);all_records.append(records)
        print(f'source {i+1}/16 frozen41, supervised41, seconds={time.monotonic()-start:.2f}',flush=True)
        del result,bank,image,clean
    x=torch.cat(xs);mse=torch.cat(ys)
    records={k:torch.cat([r[k] for r in all_records]) for k in all_records[0]}
    old_stats=training_statistics(old,x,mse,records)
    head,history=train_primary(x,mse,records)
    new_stats=training_statistics(head,x,mse,records)
    save_energy(head,a.out/'energy.pt')
    training=dict(completed_utc=datetime.now(timezone.utc).isoformat(),recipe=RECIPE,derivative_loss=DERIVATIVE_LOSS,
        rows=len(x),source_pairs=16,history=history,normalization=head.normalization(),
        statistics=dict(old_T014=old_stats,new_T023A=new_stats),opened_images=opened,
        energy_sha256=sha(a.out/'energy.pt'),manifest_sha256=sha(a.manifest),validation_normal_access=False,
        jacobian_convention=JACOBIAN,matched_bank='accepted C EV2 frozen raw states, no old state_bank or clipping')
    write(a.out/'training.json',training)
    updated=json.loads(a.assets.read_bytes());updated['files']['energy']=dict(path=str(a.out/'energy.pt'),sha256=sha(a.out/'energy.pt'))
    write(a.out/'validation_assets.json',updated)
    print('TRAINING_FROZEN '+json.dumps(dict(rows=len(x),energy_sha256=training['energy_sha256'],completed_utc=training['completed_utc'])),flush=True)


if __name__=='__main__':main()
