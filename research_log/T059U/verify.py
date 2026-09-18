import argparse,hashlib,zipfile,math
import numpy as np
from scipy.ndimage import convolve1d
from ttie.natural import load_image
from research_log.T059S.verify import bilinear
from research_log.T059U.core import *

def main(out):
    torch.set_num_threads(1);f=json.loads((out/'action_freeze.json').read_bytes());actions=json.loads((out/'actions.json').read_bytes());rows=json.loads((out/'evaluation_table.json').read_bytes());result=json.loads((out/'result.json').read_bytes());opening=json.loads((out/'reference_open.json').read_bytes());assert f['utc']<opening['first_outer_reference_read_utc']
    split=json.loads((out/'split.json').read_bytes());c2=json.loads(C2.read_bytes());assert split['image_ids']==c2['image_ids']['heldout'] and split['bank_ids']==c2['bank_indices']['heldout'] and split['row_ids']==c2['row_indices']['heldout'];assert split['zero_overlap_with_E_development']
    for n,h in f['files'].items():assert sha(out/n)==h
    field=torch.load(out/'field.pt',weights_only=True,map_location='cpu');g=np.einsum('bfi,bf->bi',field['J'].numpy().astype(float),field['q'].numpy().astype(float));np.testing.assert_allclose(g,field['g'].numpy(),rtol=3e-5,atol=2e-7)
    opens=json.loads((out/'clean_opens.json').read_bytes());clean={r['image_id']:load_image(r['path']).numpy().astype(float) for r in opens};checks=[];ung=[];gated=[];acted=[]
    for a,r in zip(actions,rows):
        assert all(a[k]==r[k] for k in ['index','bank_index','state_index','image_id']) and a['state_index']==0
        t={k:v.numpy() for k,v in torch.load(out/a['file'],weights_only=True,map_location='cpu').items()};grad=t['g'].astype(float);norm=math.sqrt(math.fsum(float(x*x) for x in grad));decision=norm>=0.031453661388567547;assert decision==a['acted']==r['acted'];assert math.isclose(norm,a['norm'],rel_tol=1e-12)
        v=-.05*grad/(abs(grad)+1e-8);adam=float(np.max(abs(v.reshape(1,1,8,8)-t['v1'])));assert adam<1e-7
        assert np.array_equal(t['v_gated'],t['v1'] if decision else np.zeros_like(t['v1'])) and np.array_equal(t['y_gated'],t['y1'] if decision else t['y0'])
        y0=t['y0'].astype(float);h,w=y0.shape[-2:];kernel=np.array([1,4,6,4,1])/16;detail=y0-convolve1d(convolve1d(y0,kernel,axis=-1,mode='mirror'),kernel,axis=-2,mode='mirror');c=np.tanh(bilinear(t['v1'],h,w));render=np.where(t['mask'],np.clip(y0+c*detail,0,1),y0);err=float(np.max(abs(render-t['y1'])));assert err<1e-6
        ms=[float(np.mean((t[k].astype(float)-clean[a['image_id']])**2)) for k in ['y0','y1','y_gated']];np.testing.assert_allclose(ms,[r['mse0'],r['mse_ungated'],r['mse_gated']],rtol=1e-12,atol=0)
        ung.append(ms[1]-ms[0]);gated.append(ms[2]-ms[0]);acted.append(decision);checks.append(dict(bank_index=a['bank_index'],adam_error=adam,render_error=err,norm=norm,acted=decision))
    assert len(checks)==80 and len(clean)==16
    ung=np.array(ung);gated=np.array(gated);acted=np.array(acted);harm=ung>0;good=ung<0;coverage=float(acted.mean());recall=float((~acted)[harm].mean()) if harm.any() else None;retention=float(acted[good].mean()) if good.any() else 0
    for label,values,mask in [('ungated',ung,np.ones(80,dtype=bool)),('gated',gated,acted)]:
        expected=result['policies'][label];assert expected['improved']==int((values<0).sum()) and expected['harmed']==int((values>0).sum()) and expected['tied']==int((values==0).sum());assert expected['acted']==int(mask.sum())
        for k,v in dict(mean_A=float(values.mean()),median_A=float(np.median(values)),p90_A=float(np.quantile(values,.9)),max_harm=float(max(0,values.max()))).items():np.testing.assert_allclose(v,expected[k],rtol=1e-10,atol=1e-15)
        if mask.any():np.testing.assert_allclose(float(np.median(values[mask])),expected['median_A_acted'],rtol=1e-10,atol=1e-15)
    assert recall==result['harmful_anchor_recall'] and retention==result['beneficial_action_retention']
    if coverage<.75:label='low-norm abstention is too indiscriminate'
    elif not harm.any():label='low-norm safety mechanism is not testable on this cohort'
    elif recall<.8 or retention<.8 or int((gated>0).sum())>1:label='fixed low-norm abstention does not transfer'
    elif gated.mean()>=0 or np.median(gated[acted])>=0:label='fixed low-norm abstention does not transfer'
    else:label='fixed low-norm abstention is supported as a source-only outer safety candidate'
    assert label==result['classification']
    for n,h in f['input_hashes_before'].items():assert sha(n)==h
    for n,h in f['source_bindings'].items():assert sha(n)==h
    for c in f['jacobian_ranges']:
        with zipfile.ZipFile(c['file']) as z:
            name=next(n for n in z.namelist() if n.endswith('/data.pkl'));assert hashlib.sha256(z.read(name)).hexdigest()==c['metadata_sha256']
        with open(c['file'],'rb') as stream:
            for v in c['ranges']:stream.seek(v['offset']);assert hashlib.sha256(stream.read(v['bytes'])).hexdigest()==v['sha256']
    for r in opens:assert sha(r['path'])==r['sha256']
    atomic_json(out/'verification.json',dict(status='PASS',banks=80,images=16,checks=checks,immutable_inputs=True,exact_gated_identity_or_ungated_output=True,utc=utc()));print('INDEPENDENT_PASS_ALL80',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
