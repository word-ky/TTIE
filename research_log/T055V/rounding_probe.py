import numpy as np,torch,json,itertools
rng=np.random.default_rng(2026);rng.uniform(-30,30,(8,8));g=rng.uniform(-30,30,(8,8)).astype(np.float32);h,w=400,600
with torch.no_grad():t=torch.nn.functional.interpolate(torch.from_numpy(g)[None,None].cuda(),size=(h,w),mode='bilinear',align_corners=False).cpu().numpy()[0,0]
def coords(n):return np.maximum(((np.arange(n,dtype=np.float32)+.5).astype(np.float64)*float(np.float32(8/n))-.5).astype(np.float32),0)
y=coords(h);x=coords(w);iy=y.astype(int);ix=x.astype(int);jy=np.minimum(iy+1,7);jx=np.minimum(ix+1,7);fy=y-iy.astype(np.float32);fx=x-ix.astype(np.float32)
def fma(a,b,c):return (a.astype(np.float64)*b.astype(np.float64)+c.astype(np.float64)).astype(np.float32)
def add(a,b,c,d,order):return fma(a,b,c*d) if order==0 else fma(c,d,a*b)
for order in itertools.product([0,1],repeat=3):
 top=add(1-fx,g[iy[:,None],ix[None,:]],fx,g[iy[:,None],jx[None,:]],order[0]);bot=add(1-fx,g[jy[:,None],ix[None,:]],fx,g[jy[:,None],jx[None,:]],order[1]);z=add(1-fy[:,None],top,fy[:,None],bot,order[2]);print(order,float(np.abs(z-t).max()),int(np.count_nonzero(z!=t)))
