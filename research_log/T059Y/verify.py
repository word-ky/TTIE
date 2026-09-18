import argparse,math,statistics
from scipy.ndimage import convolve1d
from research_log.T059S.verify import bilinear
from research_log.T059Y.core import *
def independent_ssim(x,y):
    weights=np.exp(-np.arange(-5,6,dtype=float)**2/4.5);weights/=weights.sum()
    def smooth(z):return convolve1d(convolve1d(z,weights,axis=0,mode='reflect'),weights,axis=1,mode='reflect')
    mx,my=smooth(x),smooth(y)
    return float(np.mean(((2*mx*my+.0001)*(2*(smooth(x*y)-mx*my)+.0009))/((mx*mx+my*my+.0001)*(smooth(x*x)-mx*mx+smooth(y*y)-my*my+.0009))))
def quantile(values,p):
    a=sorted(values);k=(len(a)-1)*p;i=int(k);return a[i]+(k-i)*(a[min(i+1,len(a)-1)]-a[i])
def main(out):
    torch.set_num_threads(1);f=json.loads((out/'online_freeze.json').read_bytes());opening=json.loads((out/'reference_open.json').read_bytes());assert f['utc']<opening['first_reference_read_utc'];records=json.loads((out/'online_records.json').read_bytes());table=json.loads((out/'evaluation_table.json').read_bytes());result=json.loads((out/'result.json').read_bytes());cohort=json.loads(Path('research_log/T059Y/cohort.json').read_bytes());pairing=json.loads(Path('research_log/T059Y/evaluation_pairing.json').read_bytes());assert len(records)==len(table)==len(cohort['anchors'])==100
    validate_inputs(json.loads(Path('research_log/T059Y/evaluation_binding.json').read_bytes()));checks=[];maxmetric=0.;fields={}
    for r,t,c,s in zip(records,table,cohort['anchors'],pairing['selected']):
        assert r['index']==t['index']==c['index'] and r['low']==t['low']==c['low']==s['low'];assert r['optimizer_steps']==1
        b=torch.load(out/r['file'],map_location='cpu',weights_only=True);base=torch.load(r['base_file'],weights_only=True,map_location='cpu');trajectory=torch.load(r['trajectory_file'],weights_only=True,map_location='cpu');assert torch.equal(base['image'],b['y0']) and torch.equal(base['grid'],b['global_grid']) and torch.equal(base['raw'],b['global_raw']);assert torch.equal(trajectory['states'][r['selected_step']],b['global_raw']) and torch.equal(trajectory['grids'][r['selected_step']],b['global_grid']);assert all(torch.isfinite(z).all() for z in b.values())
        gate=json.loads(Path(r['decision_file']).read_bytes())['gate'];assert gate==r['gate']
        active=torch.tensor(gate['active']).float();signed=active*torch.where(torch.tensor(gate['winner'])==0,1.,-1.);expected=torch.cat((active,signed,torch.tensor(gate['evidence'],dtype=torch.float64))).float();assert torch.equal(b['x'][:12],expected);assert torch.equal(b['x'][20:],b['global_grid'].flatten())
        g=b['g'].double().numpy();np.testing.assert_allclose(b['J'].double().numpy().T@b['q'].double().numpy(),g,rtol=3e-5,atol=2e-7)
        norm=math.sqrt(math.fsum(v*v for v in g));assert math.isclose(norm,r['norm'],rel_tol=1e-12,abs_tol=1e-15)
        adam=-.05*g/(abs(g)+1e-8);ae=float(np.max(abs(adam.reshape(1,1,8,8)-b['v1'].numpy())));assert ae<1e-7
        y0=b['y0'].numpy().astype(float);h,w=y0.shape[-2:];k=np.array([1,4,6,4,1])/16;detail=y0-convolve1d(convolve1d(y0,k,axis=-1,mode='mirror'),k,axis=-2,mode='mirror');y=np.where(b['mask'].numpy(),np.clip(y0+np.tanh(bilinear(b['v1'].numpy(),h,w))*detail,0,1),y0);re=float(np.max(abs(y-b['y1'].numpy())));assert re<1e-6
        acted=norm>0 and not torch.equal(b['y0'],b['y1']);assert acted==r['acted']==t['acted']
        if r['active_regions']==0 or norm==0:assert torch.equal(b['y0'],b['y1'])
        path=Path(pairing['normal_root'])/s['normal'];assert sha(path)==s['normal_sha256'];normal=native_rgb(path).squeeze(0).permute(1,2,0).numpy().astype(float)
        for name,key in [('base','y0'),('ours','y1')]:
            im=b[key].squeeze(0).permute(1,2,0).numpy().astype(float);m=float(torch.from_numpy(im-normal).square().mean());vals=dict(mse=m,psnr=-10*math.log10(max(m,1e-12)),ssim=independent_ssim(im,normal))
            for metric,value in vals.items():
                err=abs(value-t[name+'_'+metric]);maxmetric=max(maxmetric,err);assert err<1e-11
        for metric in ['mse','psnr','ssim']:assert abs(t[metric+'_change']-(t['ours_'+metric]-t['base_'+metric]))<1e-14
        checks.append(dict(index=r['index'],adam_error=ae,renderer_error=re));fields[r['index']]={k:b[k] for k in ['x','J','q','g','v1']}
    wins=sum(r['mse_change']< -1e-12 for r in table);harm=sum(r['mse_change']>1e-12 for r in table);coverage=sum(r['acted'] for r in table);assert (wins,harm,100-wins-harm,coverage)==(result['improve'],result['harm'],result['tie'],result['active_actions'])
    for k,d in result['statistics'].items():
        a=[r[k] for r in table];v=dict(mean=math.fsum(a)/100,median=statistics.median(a),p10=quantile(a,.1),p90=quantile(a,.9))
        for n,value in v.items():assert math.isclose(value,d[n],rel_tol=1e-11,abs_tol=1e-13)
    mean=math.fsum(r['psnr_change'] for r in table)/100;median=statistics.median(r['psnr_change'] for r in table);ssim=math.fsum(r['ssim_change'] for r in table)/100
    label='T026-conditioned matched-detail activation is insufficient' if coverage<50 else 'matched-detail provides material complementary gain over T026-A' if mean>=.05 and median>0 and wins>=60 and ssim>=.001 else 'one-step matched-detail integration is practically immaterial on the opened development cohort';assert label==result['classification']
    validate_inputs({str(out/n):h for n,h in f['files'].items()});validate_inputs(f['input_hashes_before']);validate_inputs(f['source_bindings']);assert f['optimizer_steps']==100 and f['head_state_before']==f['head_state_after']
    save_tensor(out/'online_fields.pt',fields);atomic_json(out/'verification.json',dict(status='PASS',rows=100,checks=checks,metric_max_abs_error=maxmetric,classification=label,utc=utc()));print('INDEPENDENT_PASS100',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
