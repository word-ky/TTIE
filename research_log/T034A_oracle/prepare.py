"""All100 identity-WB renderer regressions and accepted artifact bindings, lows only."""
from core import *
import argparse,json
p=argparse.ArgumentParser()
for k in ['split','low-root','accepted','prior-oracle','prior-preflight','out']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();o=prior;assert o.sha(a.split)==o.SPLIT_SHA
old=json.loads((a.prior_preflight/'preflight.json').read_bytes());assert o.sha(a.prior_preflight/'preflight.json')=='84212f1b19950db7b51fd746d4f546af35330cc565f1376c76bfe284bd2e6f16'
assert o.sha(a.prior_preflight/'bound_starts.pt')==old['starts_sha256']
assert o.sha(a.prior_oracle/'freeze.json')=='e22d9cacb45cc634ca812588927f15efb0e0bed2c89faef2691178bbd268b2a2'
assert o.sha(a.prior_oracle/'per_image.csv')=='caa23530153cf45f7b70958f8eca1e1de2675da64aff7322c05eb2cc4756d3f7'
assert o.sha(a.accepted/'freeze.json')==old['accepted_freeze_sha256']
oldfreeze=json.loads((a.prior_oracle/'freeze.json').read_bytes())
for r in oldfreeze['rows']:
    for n,h in r['files'].items():assert o.sha(a.prior_oracle/f'{r["index"]:03d}'/n)==h
split=json.loads(a.split.read_bytes())['selected'];starts=torch.load(a.prior_preflight/'bound_starts.pt',weights_only=True,map_location='cpu')['starts']
allowed={str((a.low_root/r['low']).resolve()) for r in split};opened=[];original=o.Image.open
def low_only(path,*args,**kwargs):
    name=str(Path(path).resolve());assert name in allowed;opened.append(name);return original(path,*args,**kwargs)
o.Image.open=low_only;torch.set_num_threads(1);torch.manual_seed(7);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
a.out.mkdir(parents=True,exist_ok=False);rows=[];expanded=[]
for i,r in enumerate(split):
    d=a.accepted/f'{i:03d}';binding=old['rows'][i];assert r['low']==binding['low']
    for n,h in binding['files'].items():assert o.sha(d/n)==h
    assert o.sha(a.low_root/r['low'])==r['low_sha256']
    decision=json.loads((d/'decision.json').read_bytes());low=o.native(a.low_root/r['low']).cuda();base,_=o.frozen_model(decision,low);model,box=frozen_model(decision,low)
    values=[];pair=[]
    for initial in starts[i]:
        extended=extend_start(initial);pair.append(extended)
        with torch.no_grad():base.raw.copy_(initial.cuda());model.raw.copy_(extended.cuda());error=float((base(low)-model(low)).abs().max())
        assert error<=1e-6;values.append(error)
    saved=torch.load(d/'output.pt',weights_only=True,map_location='cpu')
    with torch.no_grad():error=float((model(low).cpu()-saved['image']).abs().max())
    assert error<=1e-6
    rows.append(dict(index=i,low=r['low'],files=binding['files'],identity_wb_max_abs=values,accepted_output_max_abs=error,gate=decision['gate'],box=decision['diagnostics']['action_box']))
    expanded.append(torch.stack(pair))
torch.save(dict(label=o.LABEL,starts=torch.stack(expanded)),a.out/'bound_starts.pt')
assert len(opened)==100
o.write(a.out/'preflight.json',dict(label=o.LABEL,completed_utc=o.utc(),rows=rows,starts_sha256=o.sha(a.out/'bound_starts.pt'),split_sha256=o.sha(a.split),
    accepted_freeze_sha256=o.sha(a.accepted/'freeze.json'),prior_preflight_sha256=o.sha(a.prior_preflight/'preflight.json'),prior_freeze_sha256=o.sha(a.prior_oracle/'freeze.json'),
    prior_csv_sha256=o.sha(a.prior_oracle/'per_image.csv'),all100_identity_wb_regression=True,normal_decodes=0,opened_lows=opened,
    renderer_max_abs=max(max(r['identity_wb_max_abs']+[r['accepted_output_max_abs']]) for r in rows)))
print('WB identity regression PASS100 x2; normaldecodes0',flush=True)
