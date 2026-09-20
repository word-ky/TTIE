import argparse,json,os,sys,time,zipfile
from pathlib import Path
import torch
from research_log.T062A.infer import sha,utc,write,save,tensor_hash,ROOT
from scripts.run_t036a import initialize,models
from ttie import gamma_range_ttt,common_gain_ttt
from ttie.semantic_ttt import FixedObjective
from ttie.lolv2_gamma_core import native_rgb
from research_log.T062CR2.core import trajectory
from research_log.T063C.core import choose, progress
HERE=Path('research_log/T063D')
MANIFEST=HERE/'manifest.json'
COHORT='3206ea57061f4b45164a81f105f818de6d6d15b342a77797ce9f1eaaccc52554'
ARCHIVE=Path('/media/wenchang/F/wjq/TTIE/shared/t022a/LOL-v2.zip')
LOWROOT=Path('/media/wenchang/F/wjq/TTIE/shared/t063d/low')

def main(out):
    initialize()
    lows=json.loads((HERE/'low_inputs.json').read_bytes());rows=lows
    binding=json.loads((HERE/'binding.json').read_bytes())
    for n,h in binding.items():assert sha(n)==h,n
    assets=json.loads(Path('research_log/T036A_assets.json').read_bytes());scorer,receipt,head=models(assets)
    for r in lows:assert sha(LOWROOT/r['low'])==r['low_sha256']
    selector_hash=sha(HERE/'method.json')
    torch.optim.Adam([torch.nn.Parameter(torch.zeros(1,device='cuda'))],lr=.03)
    out.mkdir(parents=True,exist_ok=False);allowed={(LOWROOT/r['low']).resolve() for r in rows};opened=[];enabled=True
    def guard(event,args):
        if not enabled or event!='open':return
        name,mode,flags=args
        if isinstance(name,int):return
        path=Path(os.fsdecode(name)).resolve();writing=flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)
        if path==out or out in path.parents:return
        if not writing and (path in allowed or path.suffix in ['.py','.pyc','.so']):
            if path in allowed:opened.append(str(path))
            return
        raise PermissionError('T063-D low-only guard: '+str(path))
    sys.addaudithook(guard);started=utc();records=[]
    try:
        write(out/'config.json',dict(task='T063-D',source_commit=os.environ['TTIE_SOURCE_COMMIT'],cohort_sha256=COHORT,source_binding=binding,assets=assets,seed=7,tf32=False,physical_gpu=1,gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,weights=[1,10,5],exposure_target=.6,pools=[4,16],T063_updates=27,T063_selector='frozen normalized objective progress earliest crossing',rho=0.9857470621423519,selector_sha256=selector_hash,control_updates=40,control_selector='minimum T014 energy earliest tie',optimizer='Adam lr=.03',initialization='independent raw-low identity per path',started_utc=started))
        for i,(r,l) in enumerate(zip(rows,lows)):
            begin=time.perf_counter();low=native_rgb(LOWROOT/r['low']).cuda();directory=out/f'{i:03d}';directory.mkdir();outputs={'identity':low.cpu()};methods={}
            # Candidate receives only low/scorer/gate; control outputs are never arguments.
            for name,module in [('T026',gamma_range_ttt),('T036',common_gain_ttt)]:
                result,t,decision=module.trajectory(low,scorer,receipt,head,basis='region2',max_steps=40)
                outputs[name]=result['image'].clone();t.pop('images');save(directory/(name+'_trace.pt'),dict(trace=t,decision=decision))
                methods[name]=dict(selected_step=decision['selected_step'],updates=t['diagnostics']['steps'],output_hash=tensor_hash(outputs[name]),trace_sha256=sha(directory/(name+'_trace.pt')))
                del result,t
            gate=FixedObjective(scorer,low,receipt);t=trajectory(low,gate);images=t.pop('images');k=choose(t['values'],0.9857470621423519);t['selected_step']=k
            best,reduction,q=progress(t['values']);t.update(objective_best=best,objective_reduction=reduction,normalized_progress=q.tolist(),rho=0.9857470621423519)
            outputs['T063']=images[k].clone();del images
            save(directory/'T063_trace.pt',t);methods['T063']=dict(selected_step=k,updates=27,output_hash=tensor_hash(outputs['T063']),trace_sha256=sha(directory/'T063_trace.pt'))
            save(directory/'outputs.pt',outputs)
            record=dict(index=i,low=r['low'],low_sha256=l['low_sha256'],methods=methods,outputs_sha256=sha(directory/'outputs.pt'),identity_hash=tensor_hash(outputs['identity']),completed_utc=utc(),seconds=time.perf_counter()-begin)
            records.append(record);write(directory/'receipt.json',record);print(json.dumps(dict(completed=i+1,seconds=record['seconds'])),flush=True)
            del outputs,t,low
        assert len(records)==100 and len(opened)==100
        write(out/'freeze.json',dict(rows=records,outputs=300,cohort_sha256=COHORT,config_sha256=sha(out/'config.json'),started_utc=started,completed_utc=utc(),opened_low_paths=opened,normal_image_opens=0,official_test_access=0))
        print('ALL_300_FROZEN',sha(out/'freeze.json'),flush=True)
    finally:
        # Repair observed T062-A finalizer failure after scientific processing ends.
        enabled=False
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);main(a.parse_args().out.resolve())
