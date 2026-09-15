"""Read only80 authorized source targets after the complete learned-gradient freeze."""
from core import *
import argparse,time
from PIL import Image
from ttie.natural import load_image
from ttie.common_gain import CommonRegion2
p=argparse.ArgumentParser()
for k in ['stage-a','manifest','images','out']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();setup();start=time.perf_counter();began=utc()
f=json.loads((a.stage_a/'freeze.json').read_bytes());assert sha(a.manifest)==MANIFEST and f['source_manifest_sha256']==MANIFEST
assert f['probes']==22038 and f['source_target_opens']==f['optimizer_updates']==f['selection_changes']==0
for name,h in f['source_binding'].items():assert sha(name)==h,name
for row in f['rows']:assert sha(a.stage_a/row['file'])==row['sha256']
manifest=json.loads(a.manifest.read_bytes());sources={r['image_id']:r for r in manifest['images'] if r['split']=='train_t014_sobolev'};assert set(sources)==set(f['source_ids'])
allowed={str((a.images/r['filename']).resolve()) for r in sources.values()};opened=[];original=Image.open
def open_source(path,*args,**kwargs):
    name=str(Path(path).resolve());assert name in allowed;opened.append(dict(path=name,utc=utc()));return original(path,*args,**kwargs)
Image.open=open_source;clean_cache={};a.out.mkdir(parents=True,exist_ok=False);rows=[];vectors=[]
for bound in f['rows']:
    saved=torch.load(a.stage_a/bound['file'],map_location='cpu',weights_only=True);low=saved['low'].cuda();image_id=bound['image_id']
    if image_id not in clean_cache:
        item=sources[image_id];path=a.images/item['filename'];assert sha(path)==item['sha256'];clean_cache[image_id]=load_image(path).cuda()
    clean=clean_cache[image_id];model=CommonRegion2(torch.tensor(saved['gate']['active'],dtype=torch.bool)).cuda()
    for j,record in enumerate(saved['records']):
        raw=saved['raws'][j]
        with torch.no_grad():model.raw.copy_(raw.cuda())
        version=model.raw._version;output=model(low);assert thash(output)==record['output_sha256'],'SOURCE_FROZEN_OUTPUT_MISMATCH'
        loss=(output.double()-clean.double()).square().mean();g_r,=torch.autograd.grad(loss,model.raw);g_r=g_r.cpu();g_e=saved['g_e'][j]
        assert torch.isfinite(g_r).all() and torch.isfinite(g_e).all() and model.raw._version==version and torch.equal(model.raw.detach().cpu(),raw) and model.raw.grad is None
        rows.append(dict(bank_index=bound['index'],image_id=image_id,condition=bound['condition'],**record,groups={k:alignment(g_e,g_r,v) for k,v in saved['masks'].items()}))
        vectors.append(dict(bank_index=bound['index'],state_index=record['state_index'],gain=record['gain'],active=saved['gate']['active'],g_e=g_e.flatten().tolist(),g_r=g_r.flatten().tolist()))
    print(f'{bound["index"]+1}/400 source-reference banks',flush=True)
assert len(rows)==22038 and len(opened)==80 and all(r['utc']>f['completed_utc'] for r in opened)
summary=dict(label=LABEL,canonical_states=7346,probes=22038,source_ids=80,overall=aggregate(rows),by_gain={str(g):aggregate([r for r in rows if r['gain']==g]) for g in GAINS})
summary['classification']=classify(summary['overall']);write(a.out/'states.json',rows);write(a.out/'vectors.json',vectors);write(a.out/'summary.json',summary)
for row in f['rows']:assert sha(a.stage_a/row['file'])==row['sha256']
write(a.out/'receipt.json',dict(label=LABEL,started_utc=began,completed_utc=utc(),stage_a_freeze_sha256=sha(a.stage_a/'freeze.json'),stage_a_completed_utc=f['completed_utc'],opened_source_targets=opened,probes=len(rows),output_hashes_exact=len(rows),stage_a_hashes_unchanged=True,all_finite=True,raw_unchanged_during_gradients=True,optimizer_updates=0,selection_changes=0,lolv2_image_access=False,official_test_access=False,seconds=time.perf_counter()-start,files={n:sha(a.out/n) for n in ['states.json','vectors.json','summary.json']}))
print(json.dumps(summary),flush=True)
