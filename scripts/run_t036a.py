"""Matched low-only T026/common-gain qualification, freeze before references."""
import argparse,json,time,hashlib
from pathlib import Path
from datetime import datetime,timezone
import torch
from PIL import Image
from ttie.lolv2_gamma_core import native_rgb,low_image_opener,save_episode
from ttie.clip_signal import FrozenCLIP
from ttie.learned_prototypes import Prototypes
from ttie.semantic_ttt import SemanticScorer
from ttie.energy_model import load_energy
from ttie import gamma_range_ttt,common_gain_ttt
from ttie.common_gain import physical_grid

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def utc():return datetime.now(timezone.utc).isoformat()
def initialize():
    torch.manual_seed(7);torch.set_num_threads(1);torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available()
def models(assets):
    files=assets['files']
    for v in files.values():assert sha(v['path'])==v['sha256']
    encoder=FrozenCLIP.from_checkpoint(files['clip']['path'],'cuda:0')
    saved=torch.load(files['prototypes']['path'],map_location='cuda:0',weights_only=True)
    return SemanticScorer(encoder,Prototypes(saved['raw'])),json.loads(Path(files['gate']['path']).read_bytes()),load_energy(files['energy']['path']).cuda()

def main():
    p=argparse.ArgumentParser()
    for k in ['low-root','manifest','assets','preflight','out']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();initialize();pre=json.loads(a.preflight.read_bytes())
    assert pre['identity_renderer_max_abs']<=1e-6 and pre['baseline_reproduction_max_abs']<=1e-6 and pre['normal_decodes']==0
    manifest=json.loads(a.manifest.read_bytes());rows=manifest['selected'];assert len(rows)==100
    assets=json.loads(a.assets.read_bytes());assert sha(a.assets)==pre['assets_sha256']
    for r in rows:assert sha(a.low_root/r['low'])==r['low_sha256']
    opened=[];allowed={str((a.low_root/r['low']).resolve()) for r in rows};Image.open=low_image_opener(allowed,opened)
    scorer,gate,head=models(assets);a.out.mkdir(parents=True,exist_ok=False)
    write(a.out/'config.json',dict(task='T036-A',assets=assets,manifest_sha256=sha(a.manifest),preflight_sha256=sha(a.preflight),
        source_binding=pre['source_binding'],seed=7,tf32=False,max_steps=40,optimizer='Adam lr=.03',checkpoint='minimum predicted energy, earliest tie',
        accepted_source='b2359721c89db732d17e03be273e0bdb71bb377a',gain='one scalar per Region2, exp(log2*tanh), RGB shared, identity1, bounds.5..2',
        created_utc=utc(),gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda))
    frozen=[]
    for i,r in enumerate(rows):
        low=native_rgb(a.low_root/r['low']).cuda();parent=a.out/f'{i:03d}';parent.mkdir();records={}
        for name,module in [('baseline',gamma_range_ttt),('common',common_gain_ttt)]:
            torch.cuda.synchronize();start=time.perf_counter()
            result,t,decision=module.trajectory(low,scorer,gate,head,basis='region2',max_steps=40)
            torch.cuda.synchronize();seconds=time.perf_counter()-start
            d=parent/name;files=save_episode(d,result,t,decision)
            raw=t['states'][:,0]
            if name=='baseline':raw=torch.cat([raw,raw.new_zeros(len(raw),1,2,2)],1)
            fields=physical_grid(raw)[:,[0,1,2]]
            torch.save(dict(fields=fields,selected=fields[decision['selected_step']]),d/'fast_fields.pt')
            files['fast_fields.pt']=dict(sha256=sha(d/'fast_fields.pt'),bytes=(d/'fast_fields.pt').stat().st_size)
            records[name]=dict(seconds=seconds,updates=t['diagnostics']['steps'],states=len(t['states']),selected_step=decision['selected_step'],files=files)
            del result,t
        frozen.append(dict(index=i,low=r['low'],methods=records,persisted_utc=utc()))
        write(a.out/'progress.json',frozen);print(f'{i+1}/100 both methods frozen',flush=True)
    assert opened==[str((a.low_root/r['low']).resolve()) for r in rows]
    for v in assets['files'].values():assert sha(v['path'])==v['sha256']
    write(a.out/'freeze.json',dict(completed_utc=utc(),rows=frozen,opened_images=opened,normal_decodes=0,
        manifest_sha256=sha(a.manifest),config_sha256=sha(a.out/'config.json'),assets_unchanged=True))
    print('All200 low-only outputs/decisions/trajectories frozen; normaldecodes0',flush=True)
if __name__=='__main__':main()
