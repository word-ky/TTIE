"""Reproduce the accepted T036 procedure contract on synthetic pixels only."""
import json,hashlib,sys
from pathlib import Path
from datetime import datetime,timezone
here=Path(__file__).resolve().parent
release=Path('/home/wenchang/asdasdsad/wjq/TTIE/releases/20260915-035113-ttie-t036a-common')
bindings=json.loads((here/'source_binding.json').read_bytes())
for name,b in bindings.items():
    assert hashlib.sha256((release/name).read_bytes()).hexdigest()==b['sha256'],name
    assert hashlib.sha256((here/b['pinned_file']).read_bytes()).hexdigest()==b['sha256'],name
sys.path[:0]=[str(release),str(release/'tests')]
import torch
from ttie.common_gain_ttt import trajectory
from ttie.energy_model import EnergyHead
from test_semantic_ttt import scorer,RECEIPT
torch.manual_seed(7);torch.set_num_threads(1)
low=torch.full((1,3,16,18),.1);head=EnergyHead().eval().requires_grad_(False)
result,t,decision=trajectory(low,scorer(),RECEIPT,head,max_steps=40)
gradient=torch.tensor(t['diagnostics']['gradient_vectors'][0])
facts=dict(utc=datetime.now(timezone.utc).isoformat(),source='f80cea4c9d8186e0c4a0404b28ccd58c5e1b5678',source_hashes_verified=True,input='synthetic constant RGB .1,16x18; synthetic scorer fixture and random head,seed7',initial_raw_shape=list(t['states'][0].shape),initial_raw_max_abs=float(t['states'][0].abs().max()),parameter_count=t['diagnostics']['parameter_count'],steps=t['diagnostics']['steps'],first_gradient_channel_norms=[float(gradient[:,i].double().norm()) for i in range(3)],selected_step=decision['selected_step'],minimum_energy_step=min(range(len(decision['scores'])),key=lambda i:(decision['scores'][i],i)),real_image_reads=0,normal_reference_reads=0,real_experiment_runs=0)
(here/'contract_evidence.json').write_text(json.dumps(facts,indent=2)+'\n')
print(json.dumps(facts))
