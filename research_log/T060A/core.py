from research_log.T059S.core import *
import numpy as np
import ast
from types import SimpleNamespace
pinned=Path(__file__).with_name('pinned_t051_core.py.txt');assert sha(pinned)=='0a75c8b083239bc5b4c242cdd83e2e19abd75f6f14771033cfb5584594c8cbd7'
tree=ast.parse(pinned.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Spatial');ev=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='ev');fw=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward');z=fw.body[0];assert isinstance(z,ast.Assign) and z.targets[0].id=='z'
forward=ast.FunctionDef(name='forward',args=fw.args,body=[z,ast.Return(value=ast.Name(id='z',ctx=ast.Load()))],decorator_list=[]);module=ast.fix_missing_locations(ast.Module(body=[ev,forward],type_ignores=[]));scope={'torch':torch};exec(compile(module,'literal_T051_exposure','exec'),scope)
class Exposure(torch.nn.Module):
    def __init__(self,base,grid):
        super().__init__();self.register_buffer('base',base.detach().clone());self.register_buffer('low',base.detach().clone());self.register_buffer('grid',grid.detach().clone());self.u=torch.nn.Parameter(base.new_zeros(1,1,8,8))
    ev=scope['ev']
    forward=scope['forward']
def validate(inputs):
    for n,h in inputs.items():assert sha(n)==h,n
def firewall(allowed,out):
    import sys,os
    allowed={str(Path(n).resolve()) for n in allowed};out=str(out.resolve());root=str(ROOT)
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        name=str(Path(os.fsdecode(args[0])).resolve())
        if name.startswith(root+'/') and not(name in allowed or name.startswith(out+'/') or name.startswith(root+'/.venv/')):raise PermissionError('T060 target-free boundary: '+name)
    sys.addaudithook(audit)
def classification(n,positive,median):
    if n<72:return 'spatial-exposure projection is insufficiently active'
    return 'frozen energy transfers useful spatial-exposure direction on source outer cohort' if positive>=.80 and median>=.40 else 'frozen energy does not transfer spatial-exposure direction under the fixed audit'
def summaries(rows):
    eligible=[r for r in rows if r['nondegenerate']];n=len(eligible);positive=sum(r['positive_dot'] for r in eligible)/n if n else 0.;cos=[r['cosine'] for r in eligible];median=float(np.median(cos)) if n else None
    def stats(values):return dict(mean=float(np.mean(values)),median=float(np.median(values)),p10=float(np.quantile(values,.1)),p90=float(np.quantile(values,.9)),min=float(min(values)),max=float(max(values))) if values else None
    return dict(rows=len(rows),nondegenerate=n,positive_dot_fraction=positive,cosine=stats(cos),norms={k:dict(all80=stats([r[k] for r in rows]),nondegenerate=stats([r[k] for r in eligible])) for k in ['predicted_norm','reference_norm']},dot=stats([r['dot'] for r in eligible]),classification=classification(n,positive,median if median is not None else 0.))
