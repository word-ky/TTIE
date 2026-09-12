"""Diagnose first source feature-cache mismatch, using only already read source state."""
import json
from pathlib import Path
import torch
from ttie.clip_signal import FrozenCLIP
from ttie.learned_prototypes import Prototypes
from ttie.semantic_ttt import SemanticScorer,FixedObjective,Region2
from ttie.energy_model import features
from ttie.natural import load_image,degrade

root=Path('/home/wenchang/asdasdsad/wjq/TTIE')
run=root/'runs/20260912-180228-ttie-t014-stage-a/artifacts/audit'
torch.manual_seed(7);torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
config=json.loads((run/'config.json').read_text());receipt=config['frozen_receipt']
encoder=FrozenCLIP.from_checkpoint(config['model_identity']['path'],'cuda:0')
saved=torch.load('research_log/remote_runs/20260912-071126-ttie-t006-a6000/artifacts/audit/prototypes.pt',map_location='cuda:0',weights_only=True)
scorer=SemanticScorer(encoder,Prototypes(saved['raw']))
image=degrade(load_image(root/'shared/t008/val2017/000000036660.jpg').cuda(),'homogeneous_dark')
obj=FixedObjective(scorer,image,receipt);model=Region2(obj.active).to(image)
bank=torch.load(run/'training_bank/001/bank.pt',map_location='cpu',weights_only=True)
with torch.no_grad():model.raw.copy_(bank['states'][0].to(image))
def forward():return features(obj,scorer(model(image)),model.physical_grid()[:,:2]).detach().cpu()
with torch.no_grad():nograd=forward()
grad=forward();repeat=forward();stored=bank['features'][0]
result=dict(condition='homogeneous_dark',bank_row=0,active=obj.active.tolist(),stored=stored.tolist(),no_grad=nograd.tolist(),grad=grad.tolist(),
    stored_no_grad_max=float((stored-nograd).abs().max()),stored_grad_max=float((stored-grad).abs().max()),
    grad_repeat_max=float((grad-repeat).abs().max()),grad_repeat_equal=torch.equal(grad,repeat),
    failure_indices=torch.nonzero(~torch.isclose(stored,grad,atol=1e-6,rtol=1e-6)).flatten().tolist())
(run/'feature_cache_failure_probe.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
