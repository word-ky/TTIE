"""Real-CLIP regression on the already inspected first source dark episode."""
import json
from pathlib import Path
import torch
from ttie.clip_signal import FrozenCLIP
from ttie.learned_prototypes import Prototypes
from ttie.semantic_ttt import SemanticScorer,FixedObjective
from ttie.energy_model import EnergyHead
from ttie.energy_ttt import make_model,evaluate_energy
from ttie.sobolev_source import source_bank,source_derivatives
from ttie.sobolev_train import raw_gradient
from ttie.natural import load_image,degrade

root=Path('/home/wenchang/asdasdsad/wjq/TTIE')
failed=root/'runs/20260912-180228-ttie-t014-stage-a/artifacts/audit'
torch.manual_seed(7);torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
config=json.loads((failed/'config.json').read_text());receipt=config['frozen_receipt']
encoder=FrozenCLIP.from_checkpoint(config['model_identity']['path'],'cuda:0')
saved=torch.load('research_log/remote_runs/20260912-071126-ttie-t006-a6000/artifacts/audit/prototypes.pt',map_location='cuda:0',weights_only=True)
scorer=SemanticScorer(encoder,Prototypes(saved['raw']))
clean=load_image(root/'shared/t008/val2017/000000036660.jpg');image=degrade(clean.cuda(),'homogeneous_dark')
bank=source_bank(image,scorer,receipt)
records=source_derivatives(image,clean,scorer,receipt,bank)
assert torch.equal(records['features'],bank['features'])
obj=FixedObjective(scorer,image,receipt);model=make_model(obj,image,'region2')
head=EnergyHead().cuda().eval().requires_grad_(False)
with torch.no_grad():
    head.x_scale.copy_(torch.linspace(.5,2.,28,device='cuda'));head.y_scale.fill_(2.3)
value,_,_,_,_=evaluate_energy(model,image,obj,head)
direct=torch.autograd.grad(value,model.raw)[0].flatten()
x=records['features'][:1].cuda().requires_grad_()
cached=raw_gradient(head,head.standardized(x),x,records['jacobian'][:1].cuda())[0]
torch.testing.assert_close(cached,direct,atol=2e-6,rtol=2e-5)
old=torch.load(failed/'training_bank/001/bank.pt',map_location='cpu',weights_only=True)
result=dict(source_id=36660,condition='homogeneous_dark',states=len(bank['states']),all_cached_features_bitwise_equal=True,
    raw_state_max_abs_difference=float((old['states']-bank['states']).abs().max()),
    direct_gradient=direct.tolist(),cached_gradient=cached.tolist(),chain_rule_max_abs=float((cached-direct).abs().max()),
    chain_rule_atol=2e-6,chain_rule_rtol=2e-5,source_direction_rows=int(records['direction_mask'].sum()),
    source_jacobians_finite=bool(torch.isfinite(records['jacobian']).all()),
    source_reference_gradients_finite=bool(torch.isfinite(records['reference_gradient']).all()),
    no_head_training=True,no_calibration_evaluation=True)
(root/'research_log/T014_real_clip_repair_verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
