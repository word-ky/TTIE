"""Independent Adam, direct autograd, scalar-selection and NumPy metric replay."""
import argparse,math
from ttie.natural import load_image
from research_log.T060DR2.preflight import *
from research_log.T060B.verify import scalar_stats,q_analytic,close

def adam_replay(states,grads,gate,bounds):
    active=np.array(gate['active']).reshape(2,2);dark=np.array(gate['winner']).reshape(2,2)==0
    lo=np.stack((np.where(active & ~dark,-.5,0),np.where(active,.5,1)))[None];hi=np.stack((np.where(active & dark,2.,0),np.where(active,1.25,1)))[None]
    assert np.array_equal(lo,bounds['lower']) and np.array_equal(hi,bounds['upper'])
    with np.errstate(divide='ignore'):
        lower=np.concatenate((np.arctanh(lo[:,:1]/2),np.arctanh(np.log(lo[:,1:])/math.log(2))),axis=1)
        upper=np.concatenate((np.arctanh(hi[:,:1]/2),np.arctanh(np.log(hi[:,1:])/math.log(2))),axis=1)
    raw=np.zeros((1,3,2,2));m=np.zeros_like(raw);v=np.zeros_like(raw);error=0.;assert np.count_nonzero(states[0].numpy())==0
    for j,g in enumerate(grads):
        g=g.double().numpy();m=.9*m+.1*g;v=.999*v+.001*g*g
        raw-=.03*(m/(1-.9**(j+1)))/(np.sqrt(v/(1-.999**(j+1)))+1e-8)
        raw[:,:2]=np.clip(raw[:,:2],lower,upper);raw[:,2:,~active]=0
        target=states[j+1].double().numpy();error=max(error,float(abs(raw-target).max()));np.testing.assert_allclose(raw,target,rtol=3e-5,atol=3e-5)
    return error

def main(out):
    setup();f=json.loads((out/'freeze.json').read_bytes());ref=json.loads((out/'reference_open.json').read_bytes());assert f['completed_utc']<ref['first_source_clean_read_utc']
    assert len(f['rows'])==60 and f['optimizer_steps']==4800 and f['head_state_before']==f['head_state_after']
    for k in ['source_clean_reads','reference_gradient_reads','metric_reads','target_domain_access','official_test_access']:assert f[k]==0
    validate(f['inputs']);validate(f['source_bindings']);selection=json.loads(Path('research_log/T060B/selection.json').read_bytes())
    table=json.loads((out/'metrics.json').read_bytes());result=json.loads((out/'result.json').read_bytes());assert len(table)==60
    pre=json.loads((ROOT/'research_log/T060DR2_preflight/preflight.json').read_bytes());assert pre['status']=='PASS' and len(pre['checks'])==60 and pre['completed_utc']<f['started_utc']
    assert sha(ROOT/'research_log/T060DR2_preflight/preflight.json')==f['preflight_sha256'];validate(pre['inputs'])
    for r in pre['checks']:
        path=ROOT/'research_log/T060DR2_preflight'/r['gradient_file'];assert sha(path)==r['gradient_sha256'];saved=torch.load(path,weights_only=True,map_location='cpu')
        x=saved['gain'].double().numpy();y=saved['frozen_gain'].double().numpy();nx=math.sqrt(float(np.dot(x,x)));ny=math.sqrt(float(np.dot(y,y)))
        if nx*ny:
            cos=float(np.dot(x,y))/(nx*ny);rel=math.sqrt(float(np.dot(x-y,x-y)))/ny
            assert cos>=.999 and rel<=.01;assert abs(cos-r['gradient']['cosine'])<1e-12 and abs(rel-r['gradient']['relative_l2'])<1e-12
        else:assert nx==ny==0
        assert r['actual']['active']==r['expected']['active'] and r['actual']['winner']==r['expected']['winner'] and r['errors']['scores']<=1e-5
        for j,w in enumerate(r['actual']['winner']):
            ev=(r['actual']['scores'][j][w]-selection['calibration']['tau'][w])/selection['calibration']['scale'][w]
            assert math.isclose(ev,r['actual']['evidence'][j],rel_tol=0,abs_tol=1e-14)
    scorer,receipt,h014,hE=models();estate=hE.state_dict();manifest=Path('research_log/T014_source_manifest.json');assert sha(manifest)=='4dbf8c604bc44574a5823ec8e22142c8ac39574b6103c4bbd0a69647f15a2257'
    sources={r['image_id']:r for r in json.loads(manifest.read_bytes())['images'] if r['split']=='train_t014_sobolev'};clean={};checks=[];ind=[]
    for i,(r,a,trow) in enumerate(zip(f['rows'],selection['anchors'],table)):
        assert r['order']==i==trow['order'] and r['bank_index']==a['bank_index']==trow['bank_index'] and r['image_id']==a['image_id']==trow['image_id']
        low=torch.load(Path(a['directory'])/'bank_images.pt',weights_only=True,map_location='cpu')[0].cuda();assert thash(low)==r['low_hash']==a['state_before']['y0']
        image_id=a['image_id']
        if image_id not in clean:
            s=sources[image_id];path=ROOT/'shared/t008/val2017'/s['filename'];assert sha(path)==s['sha256'];clean[image_id]=load_image(path).numpy().astype(float)
        row=dict(order=i);check=dict(order=i,methods={})
        for name in ['A','B']:
            folder=out/f'{i:03d}'/name;rec=r['methods'][name];validate({str(folder/n):h for n,h in rec['files'].items()})
            traj=torch.load(folder/'trajectory.pt',weights_only=True,map_location='cpu');ims=torch.load(folder/'images.pt',weights_only=True,map_location='cpu');trace=torch.load(folder/'gradients.pt',weights_only=True,map_location='cpu');saved=torch.load(folder/'output.pt',weights_only=True,map_location='cpu');d=json.loads((folder/'decision.json').read_bytes());gate=d['gate']
            assert len(ims)==len(traj['states'])==41 and len(trace)==d['diagnostics']['steps']==40
            selected=min(range(41),key=lambda j:(d['selection']['scores'][j],j));assert selected==rec['selected_step']==d['selection']['selected_step']
            assert torch.equal(saved['image'],ims[selected]) and torch.equal(saved['raw'],traj['states'][selected]) and torch.equal(saved['grid'],traj['grids'][selected])
            assert gate['active']==a['gate']['active'] and gate['winner']==a['gate']['winner']
            for j,g in enumerate(trace):
                assert torch.equal(g['hybrid'],torch.tensor(d['diagnostics']['gradient_vectors'][j]))
                if name=='A':assert torch.equal(g['hybrid'],g['g014'])
                else:
                    assert torch.equal(g['hybrid'][:,:2],g['g014'][:,:2]) and torch.equal(g['hybrid'][:,2:3],g['g_E_gain']) and torch.equal(g['x'],traj['features'][j])
                    np.testing.assert_allclose(g['J_gain'].double().numpy().T@g['q_E'].double().numpy(),g['g_E_gain'].flatten(),rtol=3e-5,atol=2e-6)
                    np.testing.assert_allclose(q_analytic(estate,g['x'].numpy()),g['q_E'].numpy(),rtol=3e-5,atol=2e-6)
            adam=adam_replay(traj['states'],[g['hybrid'] for g in trace],gate,d['diagnostics']['action_box'])
            m=CommonRegion2(torch.tensor(gate['active'],device='cuda:0')).cuda()
            with torch.no_grad():
                for j,x in enumerate(traj['features']):
                    assert math.isclose(float(h014(x.cuda())),d['selection']['scores'][j],rel_tol=2e-5,abs_tol=2e-5)
                    m.raw.copy_(traj['states'][j].cuda());assert torch.equal(m(low).cpu(),ims[j])
            direct=0.
            if i in [0,15,30,45,59]:
                obj=FixedObjective(scorer,low,receipt)
                for j in [0,20,39]:
                    m.raw.data.copy_(traj['states'][j].cuda());x=features(obj,scorer(m(low)),m.physical_grid()[:,:2]);g014,=torch.autograd.grad(h014(x).sum(),m.raw,retain_graph=True)
                    torch.testing.assert_close(g014.cpu(),trace[j]['g014'],rtol=3e-5,atol=2e-6)
                    if name=='B':
                        ge,=torch.autograd.grad(hE(x).sum(),m.raw);torch.testing.assert_close(ge[:,2:3].cpu(),trace[j]['g_E_gain'],rtol=3e-5,atol=2e-6);direct=max(direct,float((ge[:,2:3].cpu()-trace[j]['g_E_gain']).abs().max()))
            mses=[];psnr=[]
            for image in ims:
                delta=(image.numpy().astype(float)-clean[image_id]).ravel();mse=float(np.dot(delta,delta))/len(delta);mses.append(mse);psnr.append(-10*math.log10(mse))
            best=max(range(41),key=lambda j:(psnr[j],-j));summary=dict(selected_step=selected,selected_psnr=psnr[selected],oracle_step=best,oracle_psnr=psnr[best],regret=psnr[best]-psnr[selected])
            err=max(abs(x-y) for x,y in zip(psnr,trow[name]['psnr']));assert err<1e-10;assert max(abs(x-y) for x,y in zip(mses,trow[name]['mse']))<1e-14
            assert selected==trow[name]['selected_step'] and best==trow[name]['oracle_step'];close(summary,{k:trow[name][k] for k in summary})
            row[name]=summary;check['methods'][name]=dict(adam_max_abs=adam,direct_gain_max_abs=direct,psnr_max_abs=err)
        for k,value in [('selected_delta','selected_psnr'),('oracle_delta','oracle_psnr'),('regret_delta','regret')]:row[k]=row['B'][value]-row['A'][value];assert abs(row[k]-trow[k])<1e-10
        ind.append(row);checks.append(check)
    metrics={k:scalar_stats([r[k] for r in ind]) for k in ['selected_delta','oracle_delta','regret_delta']};counts={k:dict(win=sum(r[k]>0 for r in ind),equal=sum(r[k]==0 for r in ind),loss=sum(r[k]<0 for r in ind)) for k in metrics}
    close(metrics,result['metrics']);assert counts==result['counts']
    gates=dict(oracle_mean=metrics['oracle_delta']['mean']>=.15,oracle_median=metrics['oracle_delta']['median']>0,oracle_wins=counts['oracle_delta']['win']>=36,regret_mean=metrics['regret_delta']['mean']>=.1)
    label='T059 gain-slice creates better source trajectories but T014 scalar selection leaves material value unrealized' if all(gates.values()) else 'T014-selection mismatch is not a sufficient explanation for the T060-C-R1 near-miss'
    assert gates==result['gates'] and label==result['classification'];validate(f['inputs']);validate(f['source_bindings'])
    atomic_json(out/'verification.json',dict(status='PASS',rows=60,trajectories=120,checks=checks,preflight_checks=60,representative_indices=[0,15,30,45,59],representative_steps=[0,20,39],classification=label,completed_utc=utc()));print('INDEPENDENT_PASS60',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
