import argparse,math
import numpy as np
from scipy.ndimage import convolve1d
from research_log.T059S.verify import bilinear
from research_log.T059V.core import *
def measure(a,b):
    a=a.flatten().double().tolist();b=b.flatten().double().tolist();an=math.sqrt(math.fsum(x*x for x in a));bn=math.sqrt(math.fsum(x*x for x in b));dn=math.sqrt(math.fsum((x-y)**2 for x,y in zip(a,b)))
    return dict(max_abs=max(abs(x-y) for x,y in zip(a,b)),relative=dn/max(bn,1e-30),cosine=1. if an==bn==0 else 0. if an*bn==0 else math.fsum(x*y for x,y in zip(a,b))/(an*bn),online_norm=an,cached_norm=bn)
def main(out):
    torch.set_num_threads(1);f=json.loads((out/'online_freeze.json').read_bytes());opening=json.loads((out/'comparison_open.json').read_bytes());assert f['utc']<opening['first_cached_tensor_read_utc']
    records=json.loads((out/'online_records.json').read_bytes());table=json.loads((out/'comparison_table.json').read_bytes());result=json.loads((out/'result.json').read_bytes());selection=json.loads(Path('research_log/T059V/selection.json').read_bytes());assert [r['bank_index'] for r in records]==[r['bank_index'] for r in selection['anchors']]
    field=torch.load(U/'field.pt',weights_only=True,map_location='cpu');actions={r['bank_index']:r for r in json.loads((U/'actions.json').read_bytes())};checks=[];fields={};ok=[]
    for r,t in zip(records,table):
        a=actions[r['bank_index']];online=torch.load(out/r['file'],weights_only=True,map_location='cpu');cache=torch.load(U/a['file'],weights_only=True,map_location='cpu');pos=a['position'];targets=dict(x=field['x'][pos],J=field['J'][pos],q=field['q'][pos],g=field['g'][pos],v1=cache['v1'],y1=cache['y1'])
        for k,v in targets.items():
            for key,value in measure(online[k],v).items():assert math.isclose(value,t[k][key],rel_tol=1e-10,abs_tol=1e-13),(r['bank_index'],k,key,value,t[k][key])
        g=online['g'].numpy().astype(float);J=online['J'].numpy().astype(float);q=online['q'].numpy().astype(float);np.testing.assert_allclose(J.T@q,g,rtol=3e-5,atol=2e-7)
        adam=-.05*g/(abs(g)+1e-8);adamerr=float(np.max(abs(adam.reshape(1,1,8,8)-online['v1'].numpy())));assert adamerr<1e-7
        y0=online['y0'].numpy().astype(float);h,w=y0.shape[-2:];k=np.array([1,4,6,4,1])/16;detail=y0-convolve1d(convolve1d(y0,k,axis=-1,mode='mirror'),k,axis=-2,mode='mirror');c=np.tanh(bilinear(online['v1'].numpy(),h,w));y=np.where(online['mask'].numpy(),np.clip(y0+c*detail,0,1),y0);renderr=float(np.max(abs(y-online['y1'].numpy())));assert renderr<1e-6
        accept=t['x']['max_abs']<=.00005 and t['J']['relative']<=.002 and t['g']['cosine']>=.999 and t['g']['relative']<=.01 and t['y1']['max_abs']<=.0001;assert accept==t['passed'];ok.append(accept)
        checks.append(dict(bank_index=r['bank_index'],adam_error=adamerr,render_reconstruction_error=renderr,passed=accept));fields[r['bank_index']]={k:online[k] for k in ['x','J','q','g','v1']}
    assert len(checks)==16 and sum(ok)==result['passed'];assert result['classification']==('online target-free Jacobian bridge is reproducible on source anchors' if all(ok) else 'online target-free Jacobian bridge is not yet reproducible')
    for n,h in f['files'].items():assert sha(out/n)==h
    for n,h in f['input_hashes_before'].items():assert sha(n)==h
    for n,h in f['source_bindings'].items():assert sha(n)==h
    for n,h in result['cached_comparison_inputs'].items():assert sha(n)==h
    save_tensor(out/'online_fields.pt',fields);atomic_json(out/'verification.json',dict(status='PASS',rows=16,checks=checks,immutable_inputs=True,independent='Scalar-loop errors/cosines, NumPy chain/Adam and SciPy renderer, separate fixed bounds',utc=utc()));print('INDEPENDENT_PASS16',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
