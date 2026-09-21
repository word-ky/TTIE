"""Frozen Final-Ours inference: degraded image plus immutable global manifest."""
import argparse,json
from pathlib import Path
import torch
from research_log.T063A.common import sha,thash
from research_log.T062CR2.core import trajectory
from research_log.T066A.core import features,predict
from research_log.T063C.core import choose
from research_log.T067B.core import choices,RHO
from ttie.clip_signal import FrozenCLIP
from ttie.learned_prototypes import Prototypes
from ttie.semantic_ttt import SemanticScorer,FixedObjective
from ttie.lolv2_gamma_core import native_rgb
from research_log.T070A.manifest import environment,HERE

def select_trajectory(low,trace,model):
    table=features(low,trace['images'],trace['states'],trace['values'])
    probabilities=predict(table,model);base=choose(trace['values'],RHO)
    decision=next(c for c in choices(trace['values'],probabilities.tolist(),base) if c['lambda_value']==.875)
    return decision,table,probabilities

class FinalOurs:
    def __init__(self,manifest):
        self.manifest_path=Path(manifest);self.manifest_sha256=sha(self.manifest_path)
        self.manifest=json.loads(self.manifest_path.read_bytes())
        for path,h in self.manifest['source_binding'].items():assert sha(path)==h,path
        for item in self.manifest['assets'].values():assert sha(item['path'])==item['sha256'],item['path']
        assert self.manifest['constants']==json.loads((HERE/'constants.json').read_bytes())
        assert self.manifest['environment']==environment()
        torch.manual_seed(7);torch.set_num_threads(1);torch.use_deterministic_algorithms(False)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        files=self.manifest['assets'];encoder=FrozenCLIP.from_checkpoint(files['clip']['path'],'cuda:0')
        saved=torch.load(files['prototypes']['path'],map_location='cuda:0',weights_only=True)
        self.scorer=SemanticScorer(encoder,Prototypes(saved['raw']))
        self.gate=json.loads(Path(files['gate']['path']).read_bytes())
        self.model=json.loads(Path(files['probability_model']['path']).read_bytes())
    def __call__(self,low):
        x=low.cuda();gate=FixedObjective(self.scorer,x,self.gate);trace=trajectory(x,gate)
        decision,table,probabilities=select_trajectory(x,trace,self.model);k=decision['selected_step']
        return dict(image=trace['images'][k],state=trace['states'][k],decision=decision,trace=trace,features=table,probabilities=probabilities)

def main():
    p=argparse.ArgumentParser();p.add_argument('--low',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    model=FinalOurs(a.manifest);result=model(native_rgb(a.low));a.out.mkdir(parents=True,exist_ok=False)
    torch.save(result['image'],a.out/'output.pt')
    (a.out/'decision.json').write_text(json.dumps(dict(**result['decision'],state_hash=thash(result['state']),output_hash=thash(result['image']),manifest_sha256=model.manifest_sha256),indent=2))
if __name__=='__main__':main()
