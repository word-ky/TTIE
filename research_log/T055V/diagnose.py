"""Read-only T055 index12 arithmetic diagnosis; source-semantic CUDA cross-check."""
import torch,numpy as np,json,ast,hashlib
from pathlib import Path
base=Path('/home/wenchang/asdasdsad/wjq/TTIE');out=base/'shared/t055v';out.mkdir(exist_ok=True)
root=base/'runs/20260916-205307-ttie-t055a-oracle/artifacts/REFERENCE_ORACLE_ONLY';script=base/'releases/20260916-205152-ttie-t055a-extension/research_log/T055A_oracle/replay.py'
tree=ast.parse(script.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='interpolate');ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(script),'exec'),ns)
v=torch.load(root/'012/output.pt',weights_only=True,map_location='cpu');grid=v['v'].numpy()[0,0];h,w=v['c'].shape[-2:]
with torch.no_grad():cuda_raw=torch.nn.functional.interpolate(v['v'].cuda(),size=(h,w),mode='bilinear',align_corners=False);cuda_c=cuda_raw.tanh().cpu().numpy()[0,0];cuda_raw=cuda_raw.cpu().numpy()[0,0]
saved=v['c'].numpy()[0,0];raw=ns['interpolate'](grid,h,w);old=np.tanh(raw);loc=np.unravel_index(np.abs(old-saved).argmax(),saved.shape)
def coords(n):return np.maximum(((np.arange(n,dtype=np.float32)+.5).astype(np.float64)*float(np.float32(8/n))-.5).astype(np.float32),0)
yy=coords(h);xx=coords(w);iy=yy.astype(int);ix=xx.astype(int);jy=np.minimum(iy+1,7);jx=np.minimum(ix+1,7);fy=yy-iy.astype(np.float32);fx=xx-ix.astype(np.float32)
def fma(a,b,c):return (a.astype(np.float64)*b.astype(np.float64)+c.astype(np.float64)).astype(np.float32)
top=fma(1-fx,grid[iy[:,None],ix[None,:]],fx*grid[iy[:,None],jx[None,:]])
bot=fma(1-fx,grid[jy[:,None],ix[None,:]],fx*grid[jy[:,None],jx[None,:]])
fused=fma(1-fy[:,None],top,fy[:,None]*bot)
y,x=loc;r=dict(index=12,location=[int(y),int(x)],saved_c=float(saved[loc]),old_numpy_c=float(old[loc]),source_cuda_c=float(cuda_c[loc]),old_numpy_raw=float(raw[loc]),source_cuda_raw=float(cuda_raw[loc]),fma_numpy_raw=float(fused[loc]),fma_numpy_c=float(np.tanh(fused)[loc]),source_coordinate=[float(yy[y]),float(xx[x])],fraction=[float(fy[y]),float(fx[x])],four_raw_controls=grid[np.ix_([iy[y],jy[y]],[ix[x],jx[x]])].tolist(),old_c_max=float(np.abs(old-saved).max()),source_cuda_c_max=float(np.abs(cuda_c-saved).max()),fma_c_max=float(np.abs(np.tanh(fused)-saved).max()),fma_raw_max=float(np.abs(fused-cuda_raw).max()),original_verifier_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),torch=torch.__version__)
(out/'diagnosis.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
