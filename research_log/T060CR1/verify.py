"""Independent coordinate/Adam replay and post-freeze metric verification."""
import argparse,math
from scripts.run_t036a import models
from scripts.evaluate_t026a import pixels,independent_ssim
from ttie.lolv2_gamma_core import native_rgb
from ttie.common_gain import CommonRegion2
from ttie.semantic_ttt import FixedObjective
from ttie.energy_model import features,load_energy
from research_log.T060B.verify import scalar_stats,q_analytic,close
from research_log.T060CR1.core import *

def replay(states,traces,gate,bounds):
    active=np.array(gate['active']).reshape(2,2);dark=np.array(gate['winner']).reshape(2,2)==0
    lo=np.stack((np.where(active & ~dark,-.5,0),np.where(active,.5,1)))[None];hi=np.stack((np.where(active & dark,2.,0),np.where(active,1.25,1)))[None]
    assert np.array_equal(lo,np.array(bounds['lower'])) and np.array_equal(hi,np.array(bounds['upper']))
    with np.errstate(divide='ignore',invalid='raise'):
        lower=np.concatenate((np.arctanh(lo[:,:1]/2),np.arctanh(np.log(lo[:,1:])/math.log(2))),axis=1)
        upper=np.concatenate((np.arctanh(hi[:,:1]/2),np.arctanh(np.log(hi[:,1:])/math.log(2))),axis=1)
    raw=np.zeros((1,3,2,2));m=np.zeros_like(raw);v=np.zeros_like(raw);maxerr=0.
    assert np.count_nonzero(states[0].numpy())==0
    for i,a in enumerate(traces):
        g=a['hybrid'].double().numpy();assert torch.equal(a['hybrid'][:,:2],a['g014'][:,:2]);assert torch.equal(a['hybrid'][:,2:3],a['g_E_gain'])
        np.testing.assert_allclose(a['J_gain'].double().numpy().T@a['q_E'].double().numpy(),g[:,2:3].flatten(),rtol=3e-5,atol=2e-6)
        m=.9*m+.1*g;v=.999*v+.001*g*g;raw-=.03*(m/(1-.9**(i+1)))/(np.sqrt(v/(1-.999**(i+1)))+1e-8)
        raw[:,:2]=np.clip(raw[:,:2],lower,upper);raw[:,2:,~active]=0.
        target=states[i+1].double().numpy();maxerr=max(maxerr,float(abs(raw-target).max()));np.testing.assert_allclose(raw,target,atol=3e-5,rtol=3e-5)
    return maxerr

def main(out):
    setup();f=json.loads((out/'freeze.json').read_bytes());opening=json.loads((out/'reference_open.json').read_bytes());assert f['completed_utc']<opening['first_reference_or_baseline_read_utc'];cohort=json.loads((HERE/'cohort.json').read_bytes());manifest=json.loads((HERE/'original_manifest.json').read_bytes());table=json.loads((out/'metrics.json').read_bytes());result=json.loads((out/'result.json').read_bytes())
    assert len(f['rows'])==len(cohort['rows'])==len(manifest['selected'])==len(table)==100
    assert f['cohort_sha256']==sha(HERE/'cohort.json');validate(f['inputs']);validate(f['source_bindings']);assert sha(BASE/'freeze.json')=='46e667ded785e4f3f8ba341d40d00a7e02ca652e2778fcc7ead1750665161be4';old=json.loads((BASE/'freeze.json').read_bytes())
    scorer,receipt,head014=models(cohort['assets']);headE=load_energy(HEADS['E']);estate=headE.state_dict();checks=[];ind=[]
    for rec,c,s,t,b in zip(f['rows'],cohort['rows'],manifest['selected'],table,old['rows']):
        i=rec['index'];assert i==c['index']==t['index']==b['index'];assert rec['low']==c['low']==s['low']==t['low']==b['low'];d=out/f'{i:03d}';validate({str(d/n):h for n,h in rec['files'].items()})
        trajectory=torch.load(d/'trajectory.pt',weights_only=True,map_location='cpu');saved=torch.load(d/'output.pt',weights_only=True,map_location='cpu');trace=torch.load(d/'gradients.pt',weights_only=True,map_location='cpu');dec=json.loads((d/'decision.json').read_bytes());gate=dec['gate'];active=any(gate['active']);steps=40 if active else 0
        assert rec['updates']==len(trace)==steps and len(trajectory['states'])==steps+1;selected=min(range(steps+1),key=lambda j:(dec['selection']['scores'][j],j));assert selected==rec['selected_step']==dec['selection']['selected_step'];assert torch.equal(saved['raw'],trajectory['states'][selected])
        adam_error=replay(trajectory['states'],trace,gate,dec['diagnostics']['action_box'])
        for j,a in enumerate(trace):
            assert torch.equal(a['x'],trajectory['features'][j]);assert torch.equal(a['hybrid'],torch.tensor(dec['diagnostics']['gradient_vectors'][j]))
            np.testing.assert_allclose(q_analytic(estate,a['x'].numpy()),a['q_E'].numpy(),rtol=3e-5,atol=2e-6)
        with torch.no_grad():
            for j,x in enumerate(trajectory['features']):assert math.isclose(float(head014(x.cuda())),dec['selection']['scores'][j],rel_tol=2e-5,abs_tol=2e-5)
        low=native_rgb(Path(rec['path'])).cuda();model=CommonRegion2(torch.tensor(gate['active'],device='cuda:0')).cuda();model.raw.data.copy_(saved['raw'].cuda())
        with torch.no_grad():render=model(low).cpu() if active else low.cpu()
        assert torch.equal(render,saved['image']);assert torch.equal(model.physical_grid()[:,:2].detach().cpu(),saved['grid'])
        direct_error=0.
        if i in [0,25,50,75,99] and active:
            obj=FixedObjective(scorer,low,receipt);assert obj.active.cpu().tolist()==gate['active'] and obj.winner.cpu().tolist()==gate['winner']
            for j in [0,20,39]:
                model.raw.data.copy_(trajectory['states'][j].cuda());x=features(obj,scorer(model(low)),model.physical_grid()[:,:2]);g014,=torch.autograd.grad(head014(x).sum(),model.raw,retain_graph=True);gE,=torch.autograd.grad(headE(x).sum(),model.raw)
                torch.testing.assert_close(g014.cpu(),trace[j]['g014'],rtol=3e-5,atol=2e-6);torch.testing.assert_close(gE[:,2:3].cpu(),trace[j]['g_E_gain'],rtol=3e-5,atol=2e-6);direct_error=max(direct_error,float((gE[:,2:3].cpu()-trace[j]['g_E_gain']).abs().max()))
        normalpath=NORMAL/s['normal'];assert sha(normalpath)==s['normal_sha256'];normal=pixels(normalpath).astype(np.float32).astype(np.float64);row=dict(index=i,low=s['low']);metric_error=0.
        for name,path in [('new',d/'output.pt'),('026',BASE/f'{i:03d}'/'baseline/output.pt'),('036',BASE/f'{i:03d}'/'common/output.pt')]:
            if name!='new':assert sha(path)==b['methods']['baseline' if name=='026' else 'common']['files']['output.pt']['sha256']
            image=torch.load(path,weights_only=True,map_location='cpu')['image'][0].permute(1,2,0).numpy().astype(float);diff=(image-normal).ravel();row[name+'_psnr']=-10*math.log10(float(np.dot(diff,diff))/len(diff));row[name+'_ssim']=independent_ssim(image,normal)
        for name in ['026','036']:
            for metric in ['psnr','ssim']:row['vs'+name+'_'+metric]=row['new_'+metric]-row[name+'_'+metric]
        for key in row:
            if key not in ['index','low']:metric_error=max(metric_error,abs(row[key]-t[key]));assert abs(row[key]-t[key])<1e-11
        checks.append(dict(index=i,adam_max_abs_error=adam_error,direct_gradient_max_abs_error=direct_error,metric_max_abs_error=metric_error));ind.append(row)
    metrics={k:scalar_stats([r[k] for r in ind]) for k in ind[0] if k not in ['index','low']};close(metrics,result['metrics'])
    counts={m:dict(improve=sum(r[m]>0 for r in ind),regress=sum(r[m]<0 for r in ind),tie=sum(r[m]==0 for r in ind)) for m in ['vs026_psnr','vs026_ssim','vs036_psnr','vs036_ssim']};close(counts,result['counts']);worst=min(ind,key=lambda r:(r['vs026_psnr'],r['index']));assert result['worst_index']==worst['index'];close(result['worst_gain'],worst['vs026_psnr'])
    gates=dict(mean_psnr=metrics['vs026_psnr']['mean']>=.8,regressions=counts['vs026_psnr']['regress']<=20,worst_regression=worst['vs026_psnr']>=-3.,mean_ssim=metrics['vs026_ssim']['mean']>=.006);assert gates==result['gates'];label='T059-E gain-slice substitution meets fixed T036 finite-step criteria' if all(gates.values()) else 'T059-E gain-direction advantage does not translate into a sufficiently safe/material fixed T036 trajectory improvement';assert label==result['classification']
    validate(f['inputs']);validate(f['source_bindings']);assert f['head_state_before']==f['head_state_after']
    atomic_json(out/'verification.json',dict(status='PASS',rows=100,checks=checks,representative_indices=[0,25,50,75,99],representative_steps=[0,20,39],independent='all coordinate/chain checks, explicit MLP q, NumPy Adam/CommonBox, T014 scalar selection, exact output rendering; direct original/E autograd at15 states; independent PSNR/SSIM and scalar summaries',classification=label,utc=utc()));print('INDEPENDENT_PASS100',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
