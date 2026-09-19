"""Separate source-reference process, authorized only after global freeze."""
import argparse
from ttie.natural import load_image
from research_log.T060B.core import *

def main(out):
    setup();f=json.loads((out/'prediction_freeze.json').read_bytes());assert f['rows']>=40
    validate({str(out/n):h for n,h in f['files'].items()});validate(f['inputs']);validate(f['source_bindings'])
    first=utc();assert f['utc']<first;atomic_json(out/'reference_open.json',dict(first_source_clean_read_utc=first,prediction_freeze_utc=f['utc']))
    manifest=Path('research_log/T014_source_manifest.json');assert sha(manifest)=='4dbf8c604bc44574a5823ec8e22142c8ac39574b6103c4bbd0a69647f15a2257'
    sources={r['image_id']:r for r in json.loads(manifest.read_bytes())['images'] if r['split']=='train_t014_sobolev'};clean={};opens=[];rows=[];grads={}
    for r in json.loads((out/'predictions.json').read_bytes()):
        i=r['image_id']
        if i not in clean:
            s=sources[i];path=ROOT/'shared/t008/val2017'/s['filename'];assert sha(path)==s['sha256'];clean[i]=load_image(path).cuda();opens.append(dict(image_id=i,path=str(path),sha256=s['sha256'],utc=utc()))
        t=torch.load(out/r['file'],weights_only=True,map_location='cpu');model=Gain(t['low'].cuda(),t['raw'].cuda(),torch.tensor(r['gate']['active'],device='cuda:0',dtype=torch.bool));y=model();assert thash(y)==r['state_before']['y0'] and y.shape==clean[i].shape
        loss=(y.double()-clean[i].double()).square().mean();gR,=torch.autograd.grad(loss,model.gain);assert torch.isfinite(gR).all() and torch.count_nonzero(model.gain)==0
        ref=gR.detach().cpu().double().flatten();rn=float(ref.norm());row=dict(index=r['index'],image_id=i,bank_index=r['bank_index'],reference_norm=rn)
        for h in HEADS:
            g=t['g_'+h].double();gn=float(g.norm());dot=float(torch.dot(g,ref));row[h]=dict(norm=gn,dot=dot,cosine=dot/(gn*rn) if gn*rn else 0.,positive_dot=dot>0)
        row['nondegenerate']=rn>1e-12 and all(row[h]['norm']>1e-12 for h in HEADS);rows.append(row);grads[r['bank_index']]=gR.detach().cpu()
    save_tensor(out/'reference_gradients.pt',grads);atomic_json(out/'source_clean_reads.json',opens);atomic_json(out/'alignment_table.json',rows)
    validate({str(out/n):h for n,h in f['files'].items()});validate(f['inputs']);validate(f['source_bindings'])
    result=dict(**summarize(rows),selection_freeze=f['selection_freeze'],prediction_freeze_utc=f['utc'],first_source_clean_read_utc=first,source_clean_images=len(clean),optimizer_steps=0,target_domain_access=0,lolv2_access=0,official_test_access=0,inference_reference_leakage=0,completed_utc=utc());atomic_json(out/'result.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
