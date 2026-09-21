import argparse,json,os,platform,importlib.metadata
from pathlib import Path
import torch
from research_log.T063A.common import sha,write
HERE=Path('research_log/T070A')
def environment():
    return dict(python=platform.python_version(),packages={n:importlib.metadata.version(n) for n in ['torch','torchvision','numpy','Pillow','open_clip_torch']},cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(),cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),omp_num_threads=os.environ.get('OMP_NUM_THREADS'),mkl_num_threads=os.environ.get('MKL_NUM_THREADS'),openblas_num_threads=os.environ.get('OPENBLAS_NUM_THREADS'),mkl_threading_layer=os.environ.get('MKL_THREADING_LAYER'))
def build():
    binding=json.loads((HERE/'inference_binding.json').read_bytes())
    for path,h in binding.items():assert sha(path)==h,path
    assets=json.loads(Path('research_log/T036A_assets.json').read_bytes())['files']
    assets={k:v for k,v in assets.items() if k in ['clip','prototypes','gate']}
    model='research_log/T066A/evidence/model.json';assets['probability_model']=dict(path=model,sha256=sha(model))
    for item in assets.values():assert sha(item['path'])==item['sha256'],item['path']
    return dict(name='Final-Ours T067B lambda.875',source_commit=os.environ['TTIE_SOURCE_COMMIT'],source_binding=binding,assets=assets,constants=json.loads((HERE/'constants.json').read_bytes()),environment=environment())
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();write(a.out,build());print('FINAL_MANIFEST',sha(a.out),flush=True)
