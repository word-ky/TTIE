"""Independent NumPy/SciPy reconstruction of Adam, renderer, RGB MSE and gates."""
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import convolve1d
from ttie.natural import load_image
from research_log.T059S.core import sha,atomic_json,utc

def bilinear(v,h,w):
    def axis(n):
        z=np.maximum(((np.arange(n,dtype=np.float32)+.5).astype(np.float64)*float(np.float32(8/n))-.5).astype(np.float32),0)
        lo=np.floor(z).astype(int);return lo,np.minimum(lo+1,7),z-lo
    y0,y1,fy=axis(h);x0,x1,fx=axis(w);g=v.reshape(8,8)
    top=g[y0[:,None],x0[None,:]]*(1-fx)+g[y0[:,None],x1[None,:]]*fx
    bot=g[y1[:,None],x0[None,:]]*(1-fx)+g[y1[:,None],x1[None,:]]*fx
    return top*(1-fy[:,None])+bot*fy[:,None]

def main(out):
    torch.set_num_threads(1);freeze=json.loads((out/'action_freeze.json').read_bytes());actions=json.loads((out/'actions.json').read_bytes());rows=json.loads((out/'evaluation_table.json').read_bytes());result=json.loads((out/'result.json').read_bytes());opened=json.loads((out/'reference_open.json').read_bytes())
    assert freeze['utc']<opened['first_reference_read_utc'] and len(actions)==len(rows)==80
    for n,h in freeze['files'].items():assert sha(out/n)==h
    field=torch.load(out/'field.pt',weights_only=True,map_location='cpu');g_ind=np.einsum('bfi,bf->bi',field['J'].numpy().astype(float),field['q'].numpy().astype(float))
    np.testing.assert_allclose(g_ind,field['g'].numpy(),rtol=3e-5,atol=2e-7)
    opens=json.loads((out/'clean_opens.json').read_bytes());clean={r['image_id']:load_image(r['path']).numpy().astype(float) for r in opens};checks=[]
    for action,row in zip(actions,rows):
        assert all(action[k]==row[k] for k in ['index','bank_index','state_index','image_id'])
        t={k:v.numpy() for k,v in torch.load(out/action['file'],weights_only=True,map_location='cpu').items()}
        grad=t['g'].reshape(1,1,8,8).astype(float);v=-.05*grad/(np.abs(grad)+1e-8);verr=float(np.max(np.abs(v-t['v1'])));assert verr<1e-7
        y0=t['y0'].astype(float);h,w=y0.shape[-2:];kernel=np.array([1,4,6,4,1],dtype=float)/16
        detail=y0-convolve1d(convolve1d(y0,kernel,axis=-1,mode='mirror'),kernel,axis=-2,mode='mirror')
        coeff=np.tanh(bilinear(t['v1'],h,w));render=np.where(t['mask'],np.clip(y0+coeff*detail,0,1),y0)
        derr=float(np.max(np.abs(detail-t['detail'])));cerr=float(np.max(np.abs(coeff-t['c1'])));yerr=float(np.max(np.abs(render-t['y1'])))
        assert max(derr,cerr,yerr)<1e-6
        mse0=float(np.mean((y0-clean[row['image_id']])**2));mse1=float(np.mean((t['y1'].astype(float)-clean[row['image_id']])**2))
        np.testing.assert_allclose([mse0,mse1],[row['mse0'],row['mse1']],rtol=1e-12,atol=0)
        rel=(mse1-mse0)/mse0 if mse0 else 0;np.testing.assert_allclose(rel,row['relative_mse_change'],rtol=1e-12,atol=1e-12)
        checks.append(dict(bank_index=row['bank_index'],adam_max_error=verr,detail_max_error=derr,coefficient_max_error=cerr,render_max_error=yerr,mse0=mse0,mse1=mse1,relative_value_abs_roundoff=abs(rel-row['relative_mse_change'])))
    eligible=[r for r in rows if r['eligible']];delta=np.array([r['relative_mse_change'] for r in eligible]);wins=sum(r['mse1']<r['mse0'] for r in eligible)
    expected=result['eligible'];assert wins==expected['wins'] and len(eligible)==expected['n']
    for k,v in dict(mean_relative_change=float(delta.mean()),median_relative_change=float(np.median(delta)),p90_relative_harm=float(np.quantile(np.maximum(delta,0),.9)),max_relative_harm=float(np.maximum(delta,0).max())).items():np.testing.assert_allclose(v,expected[k],rtol=1e-12,atol=1e-12)
    supported=wins/len(eligible)>=.75 and np.mean(delta)<0 and np.median(delta)<0
    assert result['classification']==('one-step detail-direction transfer is supported as a source-only mechanism candidate' if supported else 'one-step detail-direction transfer is not supported; do not extend to multi-step detail TTT')
    for n,h in freeze['input_hashes_before'].items():assert sha(n)==h
    for r in opens:assert sha(r['path'])==r['sha256']
    atomic_json(out/'verification.json',dict(status='PASS',banks=80,images=16,independent='NumPy float64 Adam equation, SciPy separable mirror blur, NumPy bilinear/tanh/clamp, direct RGB MSE and gates',checks=checks,utc=utc()))
    print('INDEPENDENT_VERIFICATION_PASS all80',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
