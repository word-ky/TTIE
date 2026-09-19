import argparse,json,hashlib,time,os,sys
from pathlib import Path
from datetime import datetime,timezone
import torch
from ttie.lolv2_gamma_core import native_rgb
from ttie.clip_signal import FrozenCLIP
from ttie.learned_prototypes import Prototypes
from ttie.semantic_ttt import SemanticScorer,FixedObjective
from research_log.T062A.core import trajectory

ROOT=Path('/home/wenchang/asdasdsad/wjq/TTIE')
HERE=Path('research_log/T062A')
MANIFEST=Path('research_log/T036A_cohort/manifest.json')
COHORT='279c74b335999d6631d436c79ea80e9e2cccfc4b97cb7c0b992829d04cfaf40b'
def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def tensor_hash(t):return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()
def write(p,d):
    with p.open('x') as f:json.dump(d,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
def save(p,d):
    with p.open('xb') as f:torch.save(d,f);f.flush();os.fsync(f.fileno())

def main(out):
    torch.set_num_threads(1);torch.manual_seed(7);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available();assert sha(MANIFEST)==COHORT
    rows=json.loads(MANIFEST.read_bytes())['selected'];assert len(rows)==100
    assets=json.loads(Path('research_log/T036A_assets.json').read_bytes())['files']
    assets={k:v for k,v in assets.items() if k in ['clip','prototypes','gate']}
    binding=json.loads((HERE/'binding.json').read_bytes())
    for n,h in binding.items():assert sha(n)==h,n
    for v in assets.values():assert sha(v['path'])==v['sha256']
    lowroot=ROOT/'shared/t036a/low'
    lowpaths=[lowroot/r['low'] for r in rows]
    for path,r in zip(lowpaths,rows):assert sha(path)==r['low_sha256']
    encoder=FrozenCLIP.from_checkpoint(assets['clip']['path'],'cuda:0')
    prot=torch.load(assets['prototypes']['path'],weights_only=True,map_location='cuda:0')
    scorer=SemanticScorer(encoder,Prototypes(prot['raw']))
    receipt=json.loads(Path(assets['gate']['path']).read_bytes())
    # Initialize lazy optimizer/Python imports before installing the data-open guard.
    torch.optim.Adam([torch.nn.Parameter(torch.zeros(1,device='cuda'))],lr=.03)
    out.mkdir(parents=True,exist_ok=False);allowed={p.resolve() for p in lowpaths};opened=[]
    allowed.update(Path(n).resolve() for n in binding)
    def guard(event,args):
        if event!='open':return
        name,mode,flags=args
        if isinstance(name,int):return
        p=Path(os.fsdecode(name)).resolve();writing=flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)
        if p==out or out in p.parents:return
        if not writing and (p in allowed or p.suffix in ['.py','.pyc','.so']):
            if p in allowed:opened.append(str(p))
            return
        raise PermissionError('T062-A low-only read guard: '+str(p))
    sys.addaudithook(guard)
    started=utc();records=[]
    write(out/'config.json',dict(task='T062-A',weights=[1,10,5],exposure_target=.6,pools=[4,16],
        spa='mean over concatenated RGB horizontal and vertical pooled differences; no padding',
        init='identity',optimizer='Adam lr=.03',updates=40,renderer='unchanged CommonRegion2/CommonBox',
        source_binding=binding,assets=assets,cohort_sha256=COHORT,gpu=torch.cuda.get_device_name(),physical_gpu=1,started_utc=started))
    for i,(r,path) in enumerate(zip(rows,lowpaths)):
        begin=time.perf_counter();low=native_rgb(path).cuda();gate=FixedObjective(scorer,low,receipt)
        t=trajectory(low,gate);d=out/f'{i:03d}';d.mkdir();images=t.pop('images')
        save(d/'images.pt',images);save(d/'trace.pt',t)
        save(d/'output.pt',dict(image=images[t['selected_step']].clone(),low=low.cpu()))
        hashes=[tensor_hash(im) for im in images]
        record=dict(index=i,low=r['low'],low_sha256=r['low_sha256'],selected_step=t['selected_step'],
            values=t['values'],rendered_hashes=hashes,files={n:sha(d/n) for n in ['images.pt','trace.pt','output.pt']},seconds=time.perf_counter()-begin)
        records.append(record);write(d/'receipt.json',record)
        print(json.dumps(dict(completed=i+1,seconds=record['seconds'])),flush=True)
        del t,images,low
    write(out/'freeze.json',dict(rows=records,started_utc=started,completed_utc=utc(),cohort_sha256=COHORT,
        config_sha256=sha(out/'config.json'),data_reads=opened,reference_reads=0,baseline_outcome_reads=0,official_test_access=0))
    print('ALL_100_FROZEN',sha(out/'freeze.json'),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out.resolve())
