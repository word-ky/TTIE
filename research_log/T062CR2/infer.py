import argparse,json,os,sys,time,zipfile
from pathlib import Path
import torch
from research_log.T062A.infer import sha,utc,write,save,tensor_hash,ROOT
from scripts.run_t036a import initialize,models
from ttie import gamma_range_ttt,common_gain_ttt
from ttie.semantic_ttt import FixedObjective
from ttie.lolv2_gamma_core import native_rgb
from research_log.T062CR2.core import trajectory
HERE=Path('research_log/T062CR2')
MANIFEST=Path('research_log/T062CR1/manifest.json')
COHORT='e8a66a350bcce5282df2e8114355afc434bcd0f4ea69e54957ebb7618f9183c2'
ARCHIVE=Path('/media/wenchang/F/wjq/TTIE/shared/t022a/LOL-v2.zip')
LOWROOT=Path('/media/wenchang/F/wjq/TTIE/shared/t062cr2/low')

def main(out):
    initialize();assert sha(MANIFEST)==COHORT
    rows=json.loads(MANIFEST.read_bytes())['selected'];lows=json.loads((HERE/'low_inputs.json').read_bytes())
    binding=json.loads((HERE/'binding.json').read_bytes())
    for n,h in binding.items():assert sha(n)==h,n
    assets=json.loads(Path('research_log/T036A_assets.json').read_bytes());scorer,receipt,head=models(assets)
    assert sha(ARCHIVE)=='9820d8b112438d94d1f5d4d25817eee618a8cf4bc63a65cd6b19f5a98c3faefa'
    # Extract low entries only; no normal entry read or decode in this process.
    with zipfile.ZipFile(ARCHIVE) as z:
        for r,l in zip(rows,lows):
            assert r['low']==l['low'];dest=LOWROOT/r['low'];dest.parent.mkdir(parents=True,exist_ok=True)
            data=z.read('LOL-v2/Real_captured/'+r['low']);import hashlib
            assert hashlib.sha256(data).hexdigest()==l['low_sha256'];dest.write_bytes(data)
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
        raise PermissionError('T062-C-R2 low-only guard: '+str(path))
    sys.addaudithook(guard);started=utc();records=[]
    try:
        write(out/'config.json',dict(task='T062-C-R2',source_commit=os.environ['TTIE_SOURCE_COMMIT'],cohort_sha256=COHORT,source_binding=binding,assets=assets,seed=7,tf32=False,physical_gpu=1,gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,weights=[1,10,5],exposure_target=.6,pools=[4,16],T062_updates=27,T062_output=27,control_updates=40,control_selector='minimum T014 energy earliest tie',optimizer='Adam lr=.03',initialization='independent raw-low identity per path',started_utc=started))
        for i,(r,l) in enumerate(zip(rows,lows)):
            begin=time.perf_counter();low=native_rgb(LOWROOT/r['low']).cuda();directory=out/f'{i:03d}';directory.mkdir();outputs={'identity':low.cpu()};methods={}
            for name,module in [('T026',gamma_range_ttt),('T036',common_gain_ttt)]:
                result,t,decision=module.trajectory(low,scorer,receipt,head,basis='region2',max_steps=40)
                outputs[name]=result['image'].clone();t.pop('images');save(directory/(name+'_trace.pt'),dict(trace=t,decision=decision))
                methods[name]=dict(selected_step=decision['selected_step'],updates=t['diagnostics']['steps'],output_hash=tensor_hash(outputs[name]),trace_sha256=sha(directory/(name+'_trace.pt')))
                del result,t
            gate=FixedObjective(scorer,low,receipt);t=trajectory(low,gate);images=t.pop('images');outputs['T062']=images[27].clone();del images
            save(directory/'T062_trace.pt',t);methods['T062']=dict(selected_step=27,updates=27,output_hash=tensor_hash(outputs['T062']),trace_sha256=sha(directory/'T062_trace.pt'))
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
