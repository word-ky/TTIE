"""GPU label-free trajectory/features, before local frozen direction inference."""
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
from ..clip_signal import FrozenCLIP
from ..learned_prototypes import Prototypes
from ..semantic_ttt import SemanticScorer
from ..energy_model import load_energy, features
from ..energy_ttt import trajectory
from ..direction_selector import load_selector, CROSS_NAMES
from ..soft_basis.renderer import FixedCorners
from .common import CROSS, read, write, sha, now, tensor_sha, preflight


def install_reference_barrier(cohort):
    blocked = []
    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes)): return
        path = Path(args[0]).resolve(); name = path.name
        if (path in {cohort/'manifest.json', cohort/'mapping.json'} or path.suffix.lower() in ('.jpg','.jpeg','.png')
                or name in ('targets.json','candidate_metrics.json','evaluation.json','summary.json')):
            # Output writes are allowed only after selection; the selector has
            # no reason to write these reference-bearing filenames either.
            blocked.append(str(path)); raise RuntimeError('Reference access before decision freeze: '+str(path))
    sys.addaudithook(audit)
    return blocked


@torch.no_grad()
def cross_vectors(image, corners, gate, calibration, scorer):
    objective = SimpleNamespace(active=torch.tensor(gate['active'],device=image.device,dtype=torch.bool),
        winner=torch.tensor(gate['winner'],device=image.device,dtype=torch.long),
        evidence=torch.tensor(gate['evidence'],device=image.device,dtype=torch.float64),calibration=calibration)
    return [features(objective,scorer(FixedCorners(corners,candidate)(image)),corners).cpu().tolist() for candidate in CROSS]


def select(cohort, output, source, device):
    lock, code = preflight(source); cohort=cohort.resolve(); blocked=install_reference_barrier(cohort)
    frozen = read(cohort/'manifest_frozen.json'); prepared = read(cohort/'prepared.json'); inputs=cohort/'inputs'
    assert frozen['source_sha']==source and frozen['finalized_utc']<prepared['completed_utc']
    assert sha(inputs/'index.json')==prepared['inputs_sha256']
    index=read(inputs/'index.json');assert index['manifest_sha256']==frozen['manifest_sha256']==prepared['manifest_sha256']
    assert index['manifest_frozen_sha256']==sha(cohort/'manifest_frozen.json')
    for asset in lock['assets'].values():assert sha(Path(asset['path']))==asset['sha256']
    assert sha(Path(lock['clip']['path']))==lock['clip']['sha256']
    encoder=FrozenCLIP.from_checkpoint(lock['clip']['path'],device)
    saved=torch.load(lock['assets']['prototypes']['path'],map_location=device,weights_only=True)
    prototypes=Prototypes(saved['raw']);scorer=SemanticScorer(encoder,prototypes)
    energy=load_energy(lock['assets']['energy']['path']).to(device)
    output.mkdir(parents=True);started=now()
    write(output/'config.json',dict(task='T018-E',source_sha=source,source_code_sha256=code,pipeline_lock_sha256=sha(Path('research_log/T018E_pipeline_lock.json')),
        selector_receipt_sha256=lock['selector_receipt_sha256'],manifest_sha256=frozen['manifest_sha256'],input_index_sha256=sha(inputs/'index.json'),
        prepared_sha256=sha(cohort/'prepared.json'),
        cross_coordinates=CROSS,feature_schema=lock['feature_schema'],device=device,seed=7,tf32=False,max_steps=40))
    rows=[];case_hashes={}
    for item in index['episodes']:
        path=inputs/item['file'];assert sha(path)==item['sha256']
        data=torch.load(path,map_location='cpu',weights_only=True);assert set(data)=={'image'}
        image=data['image'].to(device);assert tensor_sha(image)==item['pixels_sha256']
        result,t,canonical=trajectory(image,scorer,lock['gate'],energy,basis='region2',max_steps=40)
        corners=result['grid'].to(image);vectors=cross_vectors(image,corners,t['gate'],lock['gate']['calibration'],scorer)
        directory=output/f"{item['row_index']:03d}";directory.mkdir()
        torch.save(dict(corners=corners.cpu()),directory/'state.pt')
        torch.save({k:v for k,v in t.items() if k!='images'},directory/'trajectory.pt')
        write(directory/'canonical.json',canonical)
        case_hashes[str(item['row_index'])]={name:sha(directory/name) for name in ('state.pt','trajectory.pt','canonical.json')}
        rows.append(dict(row_index=item['row_index'],input_file_sha256=item['sha256'],pixels_sha256=item['pixels_sha256'],
            corners_sha256=tensor_sha(corners),features=vectors,canonical_selected_step=canonical['selected_step']))
        print('Selected',len(rows),'/120; canonical step',canonical['selected_step'],flush=True)
        del result,t,image,data,corners
    assert len(rows)==120 and [r['row_index'] for r in rows]==list(range(120))
    write(output/'features.json',rows)
    for asset in lock['assets'].values():assert sha(Path(asset['path']))==asset['sha256']
    assert not blocked and all(p.grad is None and not p.requires_grad for p in scorer.parameters())
    write(output/'features_frozen.json',dict(started_utc=started,finalized_utc=now(),source_sha=source,
        features_sha256=sha(output/'features.json'),config_sha256=sha(output/'config.json'),case_files_sha256=case_hashes,
        selector_receipt_sha256=lock['selector_receipt_sha256'],manifest_sha256=frozen['manifest_sha256'],
        reference_access=False,blocked_reference_attempts=blocked,frozen_assets_unchanged=True,episodes=120))
    print('ALL 120 GPU FEATURE ROWS FROZEN',sha(output/'features.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cohort',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--source-sha',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    select(a.cohort,a.output,a.source_sha,a.device)
