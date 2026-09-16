"""Synthetic source-semantic verification; independent of research images/outcomes."""
import ast,numpy as np,torch,json
from pathlib import Path
p=Path(__file__).with_name('replay.py');tree=ast.parse(p.read_text(encoding='utf8'));fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='interpolate');ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(p),'exec'),ns)
rng=np.random.default_rng(2026);rows=[]
for shape in [(17,23),(400,600),(31,19)]:
 for name,g in [('random',rng.uniform(-30,30,(8,8)).astype(np.float32)),('alternating',np.tile(np.array([-24.,10.],dtype=np.float32),(8,4))),('constant',np.full((8,8),-.3,dtype=np.float32))]:
  expected=ns['interpolate'](g,*shape)
  with torch.no_grad():actual=torch.nn.functional.interpolate(torch.from_numpy(g)[None,None].cuda(),size=shape,mode='bilinear',align_corners=False);c=actual.tanh().cpu().numpy()[0,0];actual=actual.cpu().numpy()[0,0]
  row=dict(shape=shape,grid=name,raw_max=float(np.abs(expected-actual).max()),coefficient_max=float(np.abs(np.tanh(expected)-c).max()));print(json.dumps(row),flush=True);rows.append(row)
assert max(r['coefficient_max'] for r in rows)<=1e-6
result=dict(status='PASS',cases=len(rows),rows=rows,optimizer_calls=0,reference_images=0)
Path(__file__).with_name('semantic_tests.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
