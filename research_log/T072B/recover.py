"""Read-only recovery of the first attempt; cannot recover lost GPU timing."""
import json,hashlib
from pathlib import Path
import torch
from research_log.T063A.common import sha,thash,utc,write
out=Path('/media/wenchang/F/wjq/TTIE/runs/T072B-uhdll-native')
low=Path('/media/wenchang/F/wjq/TTIE/shared/t072b/1003_UHD_LL.JPG')
d=Path('research_log/T072B')
initial=json.loads((out/'ours/receipt.json').read_bytes())
for p,h in initial['source_binding'].items():assert sha(p)==h,p
metadata=json.loads((d/'pairs_manifest.json').read_bytes())
inputs=json.loads((d/'input_metadata.json').read_bytes())['files'];gts=json.loads((d/'gt_metadata.json').read_bytes())['files']
a={r['name']:r for r in inputs};b={r['name']:r for r in gts}
assert len(a)==len(b)==150 and set(a)==set(b)
assert metadata['rows']==[dict(name=n,input=a[n],gt=b[n]) for n in sorted(a)]
c=json.loads((d/'smoke_selection.json').read_bytes());acq=json.loads((d/'input_receipt.json').read_bytes())
assert low.name==min(a)==c['name'] and c['file_id']==a[low.name]['id']
assert c['declared_utc']<acq['acquired_utc'] and sha(low)==acq['sha256']==initial['input_sha256']
assert sha(d/'pairs_manifest.json')==c['pairs_manifest_sha256']
image=torch.load(out/'ours/output.pt',weights_only=True,map_location='cpu')
decision=json.loads((out/'ours/decision.json').read_bytes())
assert image.dtype==torch.float32 and tuple(image.shape)==(1,3,2160,3840) and torch.isfinite(image).all()
assert thash(image)==decision['output_hash']
from research_log.T063C.core import choose
from research_log.T067B.core import choices,RHO
selected=next(v for v in choices(decision['values'],decision['probabilities'],choose(decision['values'],RHO)) if v['lambda_value']==.875)
assert all(decision[k]==v for k,v in selected.items())
assert decision['reference_reads']==0 and decision['optimizer_updates']==27
for name in ['retinexformer','snr_aware']:assert not (out/name).exists()
assert not (out/'ours/traceback.txt').exists()
report=dict(status='PASS',classification='BLOCKED',audit_scope='Saved metadata and Ours output/decision consistency only; required final telemetry missing',pairs=150,smoke=low.name,ours_output_shape=list(image.shape),ours_output_sha256=thash(image),ours_output_file_sha256=sha(out/'ours/output.pt'),ours_decision_sha256=sha(out/'ours/decision.json'),selected_step=decision['selected_step'],k_FS=decision['k_FS'],k_rho=decision['k_rho'],completed_updates=27,reference_reads=0,model_fits=0,reruns=0,peak_gpu_memory_bytes=None,inference_seconds=None,initial_gpu_free_bytes=initial['cuda_memory_free_before'],unrun=['retinexformer','snr_aware'],source_bindings_verified=True,verified_utc=utc())
write(out/'verification.json',report);print(json.dumps(report),flush=True)
