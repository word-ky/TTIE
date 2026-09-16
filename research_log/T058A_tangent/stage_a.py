"""All canonical source-bank probes, no source-target image/label access."""
from core import *
import argparse,time
from PIL import Image
from ttie.clip_signal import FrozenCLIP
from ttie.learned_prototypes import Prototypes
from ttie.semantic_ttt import SemanticScorer
from ttie.energy_model import load_energy
from ttie.common_gain import CommonRegion2

p=argparse.ArgumentParser()
for k in ['bank-root','receipt','manifest','checkpoint','prototypes','binding','canonical-selection','out']:p.add_argument('--'+k,type=Path,required=True)
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
assert sha(a.canonical_selection)=='08227ee09f4cd428ee034d1d33cea7827037e853f2fe8b0a7efedcccfecc964c'
canonical=json.loads(a.canonical_selection.read_bytes());assert canonical['banks']==banks
initial=model_hashes();a.out.mkdir(parents=True,exist_ok=False)
write(a.out/'selection.json',dict(label=LABEL,created_utc=utc(),rule='All7346 canonical source-bank rows in accepted400-entry manifest order; duplicates retained, no metric/gradient filtering',source_ids=sorted(ids),banks=banks,states=7346,probes=7346,source_receipt_sha256=RECEIPT,source_manifest_sha256=MANIFEST,bank_manifest_sha256=BANK,checkpoint_sha256=ENERGY))
max_pixels=0.;preflight=[]
# Verify the complete canonical cohort before any derivative computation.
for index,entry in enumerate(banks):
    directory=a.bank_root/entry['directory'];saved=torch.load(directory/'bank.pt',map_location='cpu',weights_only=True)
    images=torch.load(directory/'bank_images.pt',map_location='cpu',weights_only=True);decision=json.loads((directory/'bank_decisions.json').read_bytes())
    assert decision['names'][0]=='identity';low=images[0].cuda();active=torch.tensor(decision['gate']['active'],dtype=torch.bool,device='cuda')
    legacy_model=CommonRegion2(active).cuda().requires_grad_(False)
    for j,legacy in enumerate(saved['states']):
        raw=probe_raw(legacy,active.cpu(),1.)
        with torch.no_grad():legacy_model.raw.copy_(raw.cuda());y0=legacy_model(low)
        parity=float((y0.cpu()-images[j]).abs().max());assert parity<=1e-6,'SOURCE_RECONSTRUCTION_MISMATCH'
        max_pixels=max(max_pixels,parity);preflight.append(dict(bank=index,state=j,raw_sha256=thash(raw),y0_sha256=thash(y0),max_abs=parity))
assert len(preflight)==7346
write(a.out/'preflight.json',dict(completed_utc=utc(),rows=preflight,max_abs=max_pixels,source_target_opens=0))
rows=[];count=0;fd_records=[]
for index,entry in enumerate(banks):
    directory=a.bank_root/entry['directory'];saved=torch.load(directory/'bank.pt',map_location='cpu',weights_only=True)
    images=torch.load(directory/'bank_images.pt',map_location='cpu',weights_only=True);decision=json.loads((directory/'bank_decisions.json').read_bytes())
    low=images[0].cuda();obj=objective(scorer,decision['gate'],receipt['frozen_gate_receipt']['calibration'],'cuda:0')
    legacy_model=CommonRegion2(obj.active).cuda().requires_grad_(False);records=[];raws=[];gradients=[]
    for j,legacy in enumerate(saved['states']):
        raw=probe_raw(legacy,obj.active.cpu(),1.)
        with torch.no_grad():legacy_model.raw.copy_(raw.cuda());y0=legacy_model(low);grid=legacy_model.physical_grid()[:,:2]
        assert thash(raw)==preflight[count]['raw_sha256'] and thash(y0)==preflight[count]['y0_sha256']
        model=Detail(y0,obj.active,grid);before={n:thash(v) for n,v in model.state_dict().items()}
        value,output,phi=energy(model,obj,head);g_e,=torch.autograd.grad(value,model.v)
        assert all(torch.isfinite(v).all() for v in [value,output,phi,g_e])
        assert torch.equal(output,y0) and torch.count_nonzero(model.v)==0
        if count<16:
            check=fd(model,lambda v:energy(model,obj,head,v)[0],g_e);check['index']=count;fd_records.append(check)
            write(a.out/'finite_difference_energy.json',fd_records)
            assert check['passed'],'ENERGY_FINITE_DIFFERENCE_FAILURE'
        assert before=={n:thash(v) for n,v in model.state_dict().items()} and model.v.grad is None
        assert torch.equal(legacy_model.raw.cpu(),raw)
        records.append(dict(index=count,state_index=j,state_name=decision['names'][j],energy=float(value.detach()),energy_norm=float(g_e.double().norm()),output_sha256=thash(output),detail_sha256=thash(model.detail),grid_sha256=thash(grid),raw_sha256=thash(raw)))
        raws.append(raw);gradients.append(g_e.detach().cpu());count+=1
    file=a.out/f'{index:03d}.pt'
    torch.save(dict(label=LABEL,low=low.cpu(),gate=decision['gate'],records=records,raws=torch.stack(raws),g_e=torch.stack(gradients)),file)
    rows.append(dict(index=index,image_id=entry['image_id'],condition=entry['condition'],canonical_states=entry['states'],file=file.name,sha256=sha(file)))
    write(a.out/'progress.json',dict(completed_banks=len(rows),completed_states=count,utc=utc()));print(f'{index+1}/400 banks; {count}/7346 detail gradients frozen',flush=True)
assert count==7346 and model_hashes()==initial and all(p.grad is None for m in [head,scorer] for p in m.parameters())
for n,h in binding.items():assert sha(n)==h
assert sha(a.checkpoint)==ENERGY
write(a.out/'freeze.json',dict(label=LABEL,completed_utc=utc(),rows=rows,states=count,selection_sha256=sha(a.out/'selection.json'),canonical_selection_sha256=sha(a.canonical_selection),preflight_sha256=sha(a.out/'preflight.json'),source_ids=sorted(ids),source_receipt_sha256=RECEIPT,source_manifest_sha256=MANIFEST,bank_manifest_sha256=BANK,checkpoint_sha256=ENERGY,source_binding=binding,model_parameter_hashes=initial,model_unchanged=True,normalization_reproduced_from_all7346_features=True,max_canonical_output_abs_error=max_pixels,source_target_opens=0,image_file_opens=0,optimizer_updates=0,selection_changes=0,all_finite=True,raw_unchanged_during_gradients=True,finite_difference_energy=fd_records,seconds=time.perf_counter()-start,gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda))
print('ALL7346 LEARNED DETAIL GRADIENTS FROZEN',sha(a.out/'freeze.json'),flush=True)
