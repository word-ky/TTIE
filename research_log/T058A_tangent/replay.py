"""Independent source-only renderer and scalar replay; no diagnostic helper imports."""
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import convolve1d
from ttie.common_gain import CommonRegion2

p=argparse.ArgumentParser();p.add_argument('--stage-a',type=Path,required=True);p.add_argument('--stage-b',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(1);start=time.perf_counter()
def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def thash(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
f=read(a.stage_a/'freeze.json');receipt=read(a.stage_b/'receipt.json');states=read(a.stage_b/'states.json');vectors=read(a.stage_b/'vectors.json');summary=read(a.stage_b/'summary.json')
assert sha(a.stage_a/'freeze.json')==receipt['stage_a_freeze_sha256'] and f['states']==len(states)==len(vectors)==7346
assert f['optimizer_updates']==receipt['optimizer_updates']==0 and f['source_target_opens']==0 and f['model_unchanged'] and f['raw_unchanged_during_gradients'] and receipt['raw_unchanged_during_gradients']
assert all(x['utc']>f['completed_utc'] for x in receipt['opened_source_targets']) and len(receipt['opened_source_targets'])==80
assert all(sha(Path(n))==h for n,h in f['source_binding'].items())
assert all(sha(a.stage_b/n)==h for n,h in receipt['files'].items())
pre=read(a.stage_a/'preflight.json');assert sha(a.stage_a/'preflight.json')==f['preflight_sha256'] and pre['completed_utc']<f['completed_utc']
count=0;basis_max=0.;renderer_max=0.;scalar_max=0.;checks=0;calculated=[]
def check(x,y):
    global scalar_max,checks
    if x is None or isinstance(x,(bool,str)):assert x==y
    else:
        err=abs(float(x)-float(y));scalar_max=max(scalar_max,err);assert err<=1e-10,(x,y)
    checks+=1
for bound in f['rows']:
    file=a.stage_a/bound['file'];assert sha(file)==bound['sha256'];saved=torch.load(file,map_location='cpu',weights_only=True)
    active=torch.tensor(saved['gate']['active'],dtype=torch.bool,device='cuda');legacy=CommonRegion2(active).cuda().requires_grad_(False);low=saved['low'].cuda()
    for j,record in enumerate(saved['records']):
        raw=saved['raws'][j]
        with torch.no_grad():legacy.raw.copy_(raw.cuda());y0=legacy(low);grid=legacy.physical_grid()[:,:2]
        assert thash(raw)==record['raw_sha256']==pre['rows'][count]['raw_sha256'] and thash(y0)==record['output_sha256']==pre['rows'][count]['y0_sha256'] and thash(grid)==record['grid_sha256']
        # Independent double SciPy reflect-without-edge-duplication basis.
        arr=y0.cpu().numpy();k=np.array([1,4,6,4,1],dtype=np.float64)/16
        independent=arr.astype(np.float64)-convolve1d(convolve1d(arr.astype(np.float64),k,axis=-1,mode='mirror'),k,axis=-2,mode='mirror')
        # Exact float32 basis reconstruction in NumPy for tensor hash binding.
        h,w=arr.shape[-2:];pad=np.pad(arr,((0,0),(0,0),(0,0),(2,2)),mode='reflect');horizontal=np.zeros_like(arr)
        for i,t in enumerate(k):horizontal=horizontal+pad[...,i:i+w]*np.float32(t)
        pad=np.pad(horizontal,((0,0),(0,0),(2,2),(0,0)),mode='reflect');blur=np.zeros_like(arr)
        for i,t in enumerate(k):blur=blur+pad[...,i:i+h,:]*np.float32(t)
        detail=arr-blur;assert hashlib.sha256(detail.tobytes()).hexdigest()==record['detail_sha256']
        error=float(np.max(np.abs(independent-detail)));basis_max=max(basis_max,error);assert error<=1e-6
        # At exact zero grid, align_corners=False interpolation and tanh are zero.
        ys=np.maximum((np.arange(h)+.5)*8/h-.5,0);xs=np.maximum((np.arange(w)+.5)*8/w-.5,0)
        fy=ys-np.floor(ys);fx=xs-np.floor(xs);v=np.zeros((8,8));iy=np.floor(ys).astype(int);ix=np.floor(xs).astype(int)
        top=(1-fx)*v[iy[:,None],ix[None,:]]+fx*v[iy[:,None],np.minimum(ix+1,7)[None,:]]
        bot=(1-fx)*v[np.minimum(iy+1,7)[:,None],ix[None,:]]+fx*v[np.minimum(iy+1,7)[:,None],np.minimum(ix+1,7)[None,:]]
        c=np.tanh((1-fy[:,None])*top+fy[:,None]*bot);assert np.count_nonzero(c)==0
        mask=active.cpu().numpy().reshape(2,2)[(np.arange(h)>=h//2).astype(int)[:,None],(np.arange(w)>=w//2).astype(int)[None,:]]
        out=np.where(mask,np.clip(arr+c*independent,0,1),arr);error=float(np.max(np.abs(out-arr)));renderer_max=max(renderer_max,error);assert error<=1e-6
        e=np.asarray(vectors[count]['g_e'],dtype=np.float64);r=np.asarray(vectors[count]['g_r'],dtype=np.float64)
        assert np.array_equal(e,saved['g_e'][j].numpy().ravel()) and np.isfinite(e).all() and np.isfinite(r).all()
        en=float(np.linalg.norm(e));rn=float(np.linalg.norm(r));dot=float(e@r);deg=en<=1e-12 or rn<=1e-12
        values=dict(active_count=64,energy_norm=en,reference_norm=rn,dot=dot,cosine=None if deg else dot/(en*rn),degenerate=deg,positive_dot=dot>0)
        for n,val in values.items():check(val,states[count][n])
        calculated.append(values);count+=1
    print(f'{bound["index"]+1}/400 independent replay banks',flush=True)
valid=[r for r in calculated if not r['degenerate']];cos=np.array([r['cosine'] for r in valid]);s=summary['overall']
expected=dict(count=count,nondegenerate=len(valid),degenerate=count-len(valid),cosine_mean=float(cos.mean()) if len(cos) else None,cosine_median=float(np.median(cos)) if len(cos) else None,cosine_p10=float(np.quantile(cos,.1)) if len(cos) else None,cosine_p90=float(np.quantile(cos,.9)) if len(cos) else None,positive_dot_fraction=sum(r['positive_dot'] for r in valid)/len(valid) if valid else None,energy_zero_fraction=sum(r['energy_norm']<=1e-12 for r in calculated)/count,reference_zero_fraction=sum(r['reference_norm']<=1e-12 for r in calculated)/count,either_zero_fraction=(count-len(valid))/count)
for n,val in expected.items():check(val,s[n])
for key in ['energy_norm','reference_norm','dot']:
    x=np.array([r[key] for r in calculated])
    for n,val in dict(min=x.min(),max=x.max(),mean=x.mean(),median=np.median(x),p10=np.quantile(x,.1),p90=np.quantile(x,.9)).items():check(val,s[key+'_summary'][n])
verdict='frozen-energy detail tangent supported on source' if len(valid)>0 and expected['positive_dot_fraction']>=.75 and expected['cosine_median']>=.5 else 'frozen-energy detail tangent not ready'
check(verdict,summary['classification'])
for name,fdrows in [('energy',f['finite_difference_energy']),('reference',receipt['finite_difference_reference'])]:
    assert len(fdrows)==16
    for i,row in enumerate(fdrows):
        check(row['step'],.001);check(row['central'],(row['plus']-row['minus'])/.002);check(row['abs_error'],abs(row['central']-row['autograd']));assert row['passed'] and row['abs_error']<=row['tolerance']
        direction=(np.arange(64)%2*2-1)/8;check(float(np.asarray(vectors[i]['g_e' if name=='energy' else 'g_r'])@direction),row['autograd'])
out=dict(status='PASS',states=count,scalar_checks=checks,scalar_max_abs=scalar_max,basis_max_abs=basis_max,renderer_max_abs=renderer_max,zero_grid_interpolation_exact=True,old_coordinates_exact=True,source_boundary_verified=True,optimizer_updates=0,finite_difference_states=16,classification=verdict,seconds=time.perf_counter()-start)
(a.stage_b/'independent_replay.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
