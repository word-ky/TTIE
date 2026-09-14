"""GPU re-render frozen raw states; no energy model, references or optimization."""
from core import *
import argparse,time
import torch
from PIL import Image
from ttie.semantic_ttt import Region2
from ttie.common_gain import CommonRegion2
from ttie.lolv2_gamma_core import native_rgb,low_image_opener
p=argparse.ArgumentParser()
for k in ['accepted','manifest','low-root','binding','out']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(7)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;assert torch.cuda.is_available()
assert sha(a.manifest)==COHORT and sha(a.accepted/'freeze.json')==PRIOR_FREEZE and sha(a.accepted/'metrics.csv')==PRIOR_METRICS
prior=json.loads((a.accepted/'freeze.json').read_bytes());manifest=json.loads(a.manifest.read_bytes())
assert sha(a.accepted/'config.json')==prior['config_sha256']
binding=json.loads(a.binding.read_bytes())
for name,h in binding.items():assert sha(name)==h,name
opened=[];allowed={str((a.low_root/r['low']).resolve()) for r in manifest['selected']};Image.open=low_image_opener(allowed,opened)
a.out.mkdir(parents=True,exist_ok=False);rows=[];start=time.perf_counter()
for r,item in zip(prior['rows'],manifest['selected']):
    assert r['low']==item['low'] and sha(a.low_root/item['low'])==item['low_sha256']
    low=native_rgb(a.low_root/item['low']).cuda();entry=dict(index=r['index'],low=r['low'],methods={})
    for name,record in r['methods'].items():
        old=a.accepted/f'{r["index"]:03d}'/name
        for n,h in record['files'].items():assert sha(old/n)==h['sha256']
        t=torch.load(old/'trajectory.pt',weights_only=True,map_location='cpu');saved=torch.load(old/'output.pt',weights_only=True,map_location='cpu')
        fields=torch.load(old/'fast_fields.pt',weights_only=True,map_location='cpu')['fields'];d=json.loads((old/'decision.json').read_bytes())
        assert len(t['states'])==len(d['selection']['scores'])==41
        model=(Region2 if name=='baseline' else CommonRegion2)(torch.tensor(d['gate']['active'],dtype=torch.bool)).cuda().requires_grad_(False)
        images=[];field_error=0.
        with torch.no_grad():
            for step,raw in enumerate(t['states']):
                model.raw.copy_(raw.cuda());image=model(low).cpu();images.append(image)
                grid=model.physical_grid().cpu()[:,[0,1,2]][0]
                field_error=max(field_error,float((grid-fields[step]).abs().max()))
        images=torch.stack(images);selected=d['selection']['selected_step']
        error=float((images[selected]-saved['image']).abs().max());identity_error=float((images[0]-low.cpu()).abs().max())
        assert error<=1e-6 and field_error<=1e-6 and identity_error<=1e-6
        assert torch.isfinite(images).all() and 0<=images.min()<=images.max()<=1
        dest=a.out/f'{r["index"]:03d}';dest.mkdir(exist_ok=True);file=dest/(name+'.pt')
        torch.save(dict(label=LABEL,images=images),file)
        entry['methods'][name]=dict(file=str(file.relative_to(a.out)),sha256=sha(file),states=41,selected_step=selected,
            energies=d['selection']['scores'],selected_output_max_abs=error,selected_output_bit_exact=torch.equal(images[selected],saved['image']),
            identity_max_abs=identity_error,physical_field_max_abs=field_error,prior_files=record['files'])
        del images,model,t
    rows.append(entry);write(a.out/'progress.json',dict(completed=len(rows),utc=utc()));print(f'{len(rows)}/100 reconstructed both41states',flush=True)
assert len(rows)==len(opened)==100 and sha(a.accepted/'freeze.json')==PRIOR_FREEZE
write(a.out/'freeze.json',dict(label=LABEL,completed_utc=utc(),pairs=100,states=8200,optimizer_updates=0,normal_decodes=0,
    rows=rows,opened_lows=opened,cohort_sha256=COHORT,prior_freeze_sha256=PRIOR_FREEZE,prior_metrics_sha256=PRIOR_METRICS,
    source_binding=binding,gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,seconds=time.perf_counter()-start))
print('FROZEN8200 reconstructed states; normals0, optimizerupdates0',sha(a.out/'freeze.json'),flush=True)
