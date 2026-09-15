"""All canonical source-bank probes, no source-target image/label access."""
from core import *
import argparse,time
from PIL import Image
from ttie.clip_signal import FrozenCLIP
from ttie.learned_prototypes import Prototypes
from ttie.semantic_ttt import SemanticScorer
from ttie.energy_model import load_energy
from ttie.common_gain import CommonRegion2
from ttie.common_gain_ttt import evaluate_energy
p=argparse.ArgumentParser()
for k in ['bank-root','receipt','manifest','checkpoint','prototypes','binding','out']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();setup();start=time.perf_counter()
assert sha(a.receipt)==RECEIPT and sha(a.manifest)==MANIFEST and sha(a.bank_root/'training_manifest.json')==BANK and sha(a.checkpoint)==ENERGY
receipt=json.loads(a.receipt.read_bytes());manifest=json.loads(a.manifest.read_bytes());banks=json.loads((a.bank_root/'training_manifest.json').read_bytes())
assert banks==receipt['source_records_manifest']
ids={r['image_id'] for r in receipt['split_manifests']['train_t014_sobolev']};assert len(ids)==80 and {r['image_id'] for r in banks}==ids
binding=json.loads(a.binding.read_bytes())
for n,h in binding.items():assert sha(n)==h,n
def deny(*args,**kwargs):raise AssertionError('Stage A cannot open any source clean target or image file')
Image.open=deny
features=[]
for row in banks:
    for name,h in row['files'].items():assert sha(a.bank_root/row['directory']/name)==h['sha256']
    t=torch.load(a.bank_root/row['directory']/'bank.pt',map_location='cpu',weights_only=True);assert len(t['states'])==row['states'];features.append(t['features'])
all_features=torch.cat(features);assert all_features.shape==(7346,28)
head=load_energy(a.checkpoint).cuda();buffers=head.state_dict()
assert torch.equal(all_features.double().mean(0).float(),buffers['x_mean'].cpu())
std=all_features.double().std(0,unbiased=False);assert torch.equal(torch.where(std==0,torch.ones_like(std),std).float(),buffers['x_scale'].cpu())
model_id=receipt['model_identity'];proto=receipt['frozen_gate_receipt']['prototype_identity']
assert sha(model_id['path'])==model_id['sha256'] and sha(a.prototypes)==proto['sha256']
scorer=SemanticScorer(FrozenCLIP.from_checkpoint(model_id['path'],'cuda:0'),Prototypes(torch.load(a.prototypes,map_location='cuda:0',weights_only=True)['raw'])).eval().requires_grad_(False)
def model_hashes():return {prefix+'.'+n:thash(v) for prefix,m in [('scorer',scorer),('head',head)] for n,v in m.state_dict().items()}
initial=model_hashes();a.out.mkdir(parents=True,exist_ok=False)
write(a.out/'selection.json',dict(label=LABEL,created_utc=utc(),rule='All7346 canonical source-bank rows in accepted400-entry manifest order; duplicates retained, no metric/gradient filtering',source_ids=sorted(ids),banks=banks,states=7346,gains=GAINS,probes=22038,source_receipt_sha256=RECEIPT,source_manifest_sha256=MANIFEST,bank_manifest_sha256=BANK,checkpoint_sha256=ENERGY))
rows=[];max_pixels=0.;max_features=0.;count=0
for index,entry in enumerate(banks):
    directory=a.bank_root/entry['directory'];saved=torch.load(directory/'bank.pt',map_location='cpu',weights_only=True)
    images=torch.load(directory/'bank_images.pt',map_location='cpu',weights_only=True);decision=json.loads((directory/'bank_decisions.json').read_bytes())
    assert decision['names'][0]=='identity';low=images[0].cuda();obj=objective(scorer,decision['gate'],receipt['frozen_gate_receipt']['calibration'],'cuda:0')
    model=CommonRegion2(obj.active).cuda();groups=masks(obj.active.cpu());records=[];raws=[];gradients=[];phis=[]
    for state_index,legacy in enumerate(saved['states']):
        # Establish faithful reconstruction with gain1 before probing this fixed state.
        raw=probe_raw(legacy,obj.active.cpu(),1.)
        with torch.no_grad():model.raw.copy_(raw.cuda())
        value,output,_,_,phi=evaluate_energy(model,low,obj,head)
        parity=float((output.detach().cpu()-images[state_index]).abs().max());assert parity<=1e-6,'SOURCE_RECONSTRUCTION_MISMATCH'
        ferr=float((phi.detach().cpu()-saved['features'][state_index]).abs().max());max_pixels=max(max_pixels,parity);max_features=max(max_features,ferr)
        # Cached CLIP feature drift is recorded; faithful frozen-state reconstruction
        # is established by the unchanged raw/gate and <=1e-6 pixel parity above.
        for gain in GAINS:
            raw=probe_raw(legacy,obj.active.cpu(),gain)
            assert torch.equal(raw[:,:2],legacy)
            with torch.no_grad():model.raw.copy_(raw.cuda())
            version=model.raw._version;value,output,_,_,phi=evaluate_energy(model,low,obj,head)
            g_e,=torch.autograd.grad(value,model.raw)
            assert all(torch.isfinite(v).all() for v in [raw,output,g_e,phi,value])
            assert model.raw._version==version and torch.equal(model.raw.detach().cpu(),raw) and model.raw.grad is None
            records.append(dict(state_index=state_index,state_name=decision['names'][state_index],gain=gain,energy=float(value.detach()),output_sha256=thash(output),canonical_output_max_abs=parity,canonical_feature_max_abs=ferr))
            raws.append(raw);gradients.append(g_e.cpu());phis.append(phi.detach().cpu());count+=1
    file=a.out/f'{index:03d}.pt'
    torch.save(dict(label=LABEL,low=low.cpu(),gate=decision['gate'],masks=groups,records=records,raws=torch.stack(raws),g_e=torch.stack(gradients),features=torch.stack(phis)),file)
    rows.append(dict(index=index,image_id=entry['image_id'],condition=entry['condition'],canonical_states=entry['states'],probes=len(records),file=file.name,sha256=sha(file),source_files=entry['files']))
    write(a.out/'progress.json',dict(completed_banks=len(rows),completed_probes=count,utc=utc()));print(f'{index+1}/400 banks; {count}/22038 probes frozen',flush=True)
assert count==22038 and model_hashes()==initial and all(p.grad is None for m in [head,scorer] for p in m.parameters())
assert sha(a.checkpoint)==ENERGY and sha(model_id['path'])==model_id['sha256'] and sha(a.prototypes)==proto['sha256']
write(a.out/'freeze.json',dict(label=LABEL,completed_utc=utc(),selection_sha256=sha(a.out/'selection.json'),rows=rows,canonical_states=7346,probes=count,source_ids=sorted(ids),source_receipt_sha256=RECEIPT,source_manifest_sha256=MANIFEST,bank_manifest_sha256=BANK,checkpoint_sha256=ENERGY,model_identity=model_id,prototype_sha256=proto['sha256'],source_binding=binding,model_parameter_hashes=initial,model_unchanged=True,normalization_reproduced_from_all7346_features=True,
    max_canonical_output_abs_error=max_pixels,max_canonical_feature_abs_error=max_features,source_target_opens=0,image_file_opens=0,optimizer_updates=0,selection_changes=0,all_finite=True,raw_unchanged_during_gradients=True,seconds=time.perf_counter()-start,gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda))
print('ALL SOURCE LEARNED GRADIENTS FROZEN',sha(a.out/'freeze.json'),flush=True)
