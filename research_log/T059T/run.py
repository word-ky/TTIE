import argparse,json,hashlib,datetime,os
from pathlib import Path
import torch
from research_log.T059T.core import diagnose,summarize
ROOT=Path('/home/wenchang/asdasdsad/wjq/TTIE');S=ROOT/'releases/20260918-ttie-t059s-one-step';OUT=ROOT/'runs/20260918-231247-ttie-t059s-one-step/artifacts/T059S'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(path,value):
    with path.open('w') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main(out):
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);auth=json.loads(Path('research_log/T059T/authorization.json').read_bytes());files={};own=json.loads(Path('research_log/T059T/source_binding.json').read_bytes())
    for n,h in own.items():assert sha(n)==h,n
    for n,h in auth['pinned_S_git_blobs'].items():
        data=(S/n).read_bytes();assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==h,n;files[str(S/n)]=hashlib.sha256(data).hexdigest()
    freeze=json.loads((OUT/'action_freeze.json').read_bytes())
    for n,h in freeze['files'].items():assert sha(OUT/n)==h,n;files[str(OUT/n)]=h
    for n in ['reference_gradients.pt','evaluation_table.json','action_freeze.json']:
        assert sha(OUT/n)==sha(S/'research_log/T059S'/n);files[str(OUT/n)]=sha(OUT/n)
    actions=json.loads((OUT/'actions.json').read_bytes());table=json.loads((OUT/'evaluation_table.json').read_bytes());field=torch.load(OUT/'field.pt',weights_only=True,map_location='cpu');refs=torch.load(OUT/'reference_gradients.pt',weights_only=True,map_location='cpu')
    assert len(actions)==len(table)==80 and len({r['bank_index'] for r in actions})==80 and len({r['image_id'] for r in actions})==16
    alignment=[];rows=[];compact=[]
    for action,row in zip(actions,table):
        assert all(action[k]==row[k] for k in ['index','bank_index','state_index','image_id']) and row['state_index']==0
        t=torch.load(OUT/action['file'],weights_only=True,map_location='cpu');pos=action['position'];assert int(field['global_indices'][pos])==row['index'] and torch.equal(field['g'][pos],t['g'])
        g=t['g'].flatten();v=t['v1'].flatten();ref=refs[row['index']].flatten();assert all(torch.isfinite(x).all() for x in [g,v,ref])
        eligible=bool(ref.double().norm()>1e-12);assert eligible==row['eligible'];ids={k:row[k] for k in ['index','bank_index','state_index','image_id']}
        alignment.append(dict(**ids,eligible=eligible));compact.append(dict(**ids,eligible=eligible,g=g.tolist(),v=v.tolist(),reference=ref.tolist(),mse0=row['mse0'],mse1=row['mse1']))
        if eligible:rows.append(dict(**ids,**diagnose(g.numpy(),v.numpy(),ref.numpy(),row['mse0'],row['mse1'])))
    assert len(rows)==61 and sum(r['actual_harm'] for r in rows)==4
    result=summarize(rows);result.update(counters={k:0 for k in ['new_actions','renderer_calls','optimizer_steps','model_or_feature_forwards','outer_supervision_reads','target_domain_access','lolv2_access','official_test_access','inference_reference_leakage']},completed_utc=utc(),S_evidence='27d2c2d41c6a2aebfd1b16ab78b9bf2e0ad853e3',all_anchor_rows=80)
    for n,h in files.items():assert sha(n)==h,n
    for n,h in own.items():assert sha(n)==h,n
    write(out/'alignment.json',alignment);write(out/'frozen_numeric_inputs.json',compact);write(out/'diagnostic_table.json',rows);write(out/'harmful_anchors.json',[r for r in rows if r['actual_harm']]);write(out/'result.json',result);write(out/'input_receipt.json',dict(input_hashes_before=files,input_hashes_after=files,source_bindings=own,immutable=True,utc=utc()));print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
