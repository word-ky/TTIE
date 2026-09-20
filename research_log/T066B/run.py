"""Frozen target-free table -> post-freeze reference diagnosis; no model fitting."""
import argparse,json,os,time,io,zipfile,hashlib,math
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from research_log.T063A.common import sha,write,utc,thash
from research_log.T066A.run import load_inputs,state_input,F,TARGET,CommonRegion2
from research_log.T066A.core import predict
from research_log.T066B.core import geometry,scores,distribution,classification
HERE=Path('research_log/T066B')
PRIOR=Path('research_log/T066A/evidence')


def read(name):return json.loads((PRIOR/name).read_bytes())


def transfer_labels(records,independent=False):
    raw,items,recs=load_inputs(True)
    refs=json.loads(Path('research_log/T063D/reference_inputs.json').read_bytes());rows=[]
    with zipfile.ZipFile(F/'shared/t022a/LOL-v2.zip') as z:
        for i,(item,rec,reported) in enumerate(zip(items,recs,records)):
            low,trace=state_input(raw,rec,True);x=low.cuda();model=CommonRegion2(trace['active']).to(x).eval().requires_grad_(False)
            opened=utc();data=z.read('LOL-v2/Real_captured/'+item['normal']);assert hashlib.sha256(data).hexdigest()==refs[i]['normal_sha256']
            with Image.open(io.BytesIO(data)) as im:normal=(np.asarray(im.convert('RGB'),dtype=np.float64)/255).astype(np.float32).astype(np.float64)
            savedpath=TARGET/f'{i:03d}'/'outputs.pt';assert sha(savedpath)==rec['outputs_sha256'];saved=torch.load(savedpath,weights_only=True,map_location='cpu')['T026'];assert thash(saved)==rec['methods']['T026']['output_hash']
            def psnr(image):
                if independent:
                    d=image[0].permute(1,2,0).cpu().numpy().astype(np.float64)-normal;mse=float(d.ravel()@d.ravel())/d.size
                else:
                    n=torch.as_tensor(normal,device='cuda',dtype=torch.float64);d=image[0].permute(1,2,0).double().cuda()-n;mse=float(d.square().mean())
                return -10*math.log10(mse)
            anchor=psnr(saved)
            with torch.no_grad():
                for k,state in enumerate(trace['states'][:28]):
                    model.raw.copy_(state.to(x));image=model(x);assert thash(image.cpu())==reported['hashes'][k];quality=psnr(image);margin=quality-anchor
                    rows.append(dict(index=i,low=item['low'],step=k,psnr=quality,t026_psnr=anchor,quality_margin=margin,safe=int(margin>=-5.614),reference_read_utc=opened))
            print('labels-verified' if independent else 'labels-scored',i+1,flush=True)
    return rows


def main(out):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    binding=json.loads((HERE/'binding.json').read_bytes())
    for p,h in binding.items():assert sha(p)==h,p
    out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    write(out/'config.json',dict(task='T066-B',source_commit=os.environ['TTIE_SOURCE_COMMIT'],source_binding=binding,started_utc=utc(),gpu=torch.cuda.get_device_name(),physical_gpu=1,optimizer_runs=0,model_fits=0))
    dev=read('development_features.json')['rows'];transfer=read('transfer_freeze.json')['rows'];model=read('model.json');norm=read('normalization.json')
    assert model['mean']==norm['mean'] and model['scale']==norm['scale']
    tables=[]
    for records in [dev,transfer]:
        table=[]
        for rec in records:
            probs=predict(rec['features'],model).tolist()
            if 'probabilities' in rec:np.testing.assert_array_equal(probs,rec['probabilities'])
            for k in range(28):table.append(dict(index=rec['index'],step=k,base_step=rec['base_step'],features=rec['features'][k],p_safe=probs[k],render_hash=rec['hashes'][k]))
        tables.append(table)
    dt,tt=tables;write(out/'development_target_free.json',dt);write(out/'transfer_target_free.json',tt)
    write(out/'diagnostic_freeze.json',dict(development_sha256=sha(out/'development_target_free.json'),transfer_sha256=sha(out/'transfer_target_free.json'),model_sha256=sha(PRIOR/'model.json'),normalization_sha256=sha(PRIOR/'normalization.json'),prior_transfer_sha256=sha(PRIOR/'transfer_freeze.json'),reference_reads=0,frozen_utc=utc()))
    first=utc();write(out/'reference_open.json',dict(first_reference_quality_read_utc=first,freeze_sha256=sha(out/'diagnostic_freeze.json')))
    for p,h in json.loads((HERE/'evaluation_binding.json').read_bytes()).items():assert sha(p)==h,p
    labels=transfer_labels(transfer);write(out/'transfer_labels.json',labels)
    training=read('training_table.json');dy=np.asarray([r['safe'] for r in training],dtype=bool);ty=np.asarray([r['safe'] for r in labels],dtype=bool)
    for a,b in zip(dt,training):assert a['features']==b['features'] and a['index']==b['index'] and a['step']==b['step']
    z=lambda rows:(np.asarray([r['features'] for r in rows])-np.asarray(norm['mean']))/np.asarray(norm['scale'])
    dz,tz=z(dt),z(tt);dg=geometry(dz,dz,dy,query_ids=np.repeat(np.arange(100),28),device='cuda');tg=geometry(tz,dz,dy,device='cuda')
    for table,geo in [(dt,dg),(tt,tg)]:
        for r,g in zip(table,geo):g.update(index=r['index'],step=r['step'])
    write(out/'development_geometry.json',dg);write(out/'transfer_geometry.json',tg)
    probs=np.asarray([r['p_safe'] for r in tt]);prefix=np.asarray([r['step']<=r['base_step'] for r in tt]);base=np.asarray([r['step']==r['base_step'] for r in tt]);allmask=np.ones(len(tt),dtype=bool)
    summary={name:scores(ty,probs,mask) for name,mask in [('all',allmask),('prefix',prefix),('base',base)]}
    support=distribution(tg,~ty);false_bases=int(np.sum(base&~ty&(probs>=.5)))
    tails=[]
    for j,(r,q,g) in enumerate(zip(tt,labels,tg)):
        if base[j] and (not ty[j] or r['index'] in [16,86]):tails.append(dict(index=r['index'],step=r['step'],p_safe=r['p_safe'],safe=q['safe'],quality_margin=q['quality_margin'],**{k:v for k,v in g.items() if k not in ['index','step']}))
    result=dict(classification=classification(summary['all']['unsafe_recall'],support['fraction_m_positive'],false_bases),label='diagnostic-only exposed-cohort audit; no new selector',transfer=summary,transfer_unsafe_support=support,development_loio_unsafe_support=distribution(dg,~dy),unsafe_base_rows=tails,false_safe_base_states=false_bases,transfer_unsafe_exact_distance_ties=sum(g['margin']==0 for g,m in zip(tg,~ty) if m),development_classifier_context='T066-A held-out classifier recall73/81; target-free development table here uses frozen final model',completed_utc=utc(),seconds=time.perf_counter()-started,optimizer_runs=0,model_fits=0)
    write(out/'result.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out.resolve())
