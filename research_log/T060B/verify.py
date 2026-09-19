"""Independent gain derivative, explicit MLP derivative and scalar summaries."""
import argparse,math,statistics
from ttie.natural import load_image
from research_log.T060B.core import *

def reference_analytic(low,raw,active,clean):
    # Independently spell out the fixed EV/gamma portion, without CommonRegion2
    # or autograd. Float32 forward on the same device preserves clamp boundaries.
    h,w=low.shape[-2:];yy=(torch.arange(h,device=low.device)>=h//2).long();xx=(torch.arange(w,device=low.device)>=w//2).long()
    ev=2*raw[:,:1].tanh();gamma=(math.log(2)*raw[:,1:2].tanh()).exp()
    ev=ev[:,:,yy[:,None],xx[None,:]];gamma=gamma[:,:,yy[:,None],xx[None,:]]
    pre=(low*torch.exp2(ev)+1e-6).pow(gamma)-torch.pow(1e-6,gamma)
    before_clamp=.5+(pre-.5);valid=((before_clamp>=0)&(before_clamp<=1)).cpu().numpy()
    y=before_clamp.clamp(0,1).cpu().numpy().astype(float);pre=pre.cpu().numpy().astype(float)
    pixel=(2*(y-clean)/low.numel()*pre*math.log(2)*valid).sum(axis=(0,1));g=np.zeros((2,2))
    ys=[slice(0,h//2),slice(h//2,h)];xs=[slice(0,w//2),slice(w//2,w)]
    for j in range(2):
        for k in range(2):g[j,k]=pixel[ys[j],xs[k]].sum() if active[j*2+k] else 0.
    return g.reshape(1,1,2,2)

def q_analytic(state,x):
    s={k:v.double().numpy() for k,v in state.items()};z=(np.asarray(x,dtype=float)-s['x_mean'])/s['x_scale'];derivatives=[]
    for i in [0,2]:
        a=s[f'net.{i}.weight']@z+s[f'net.{i}.bias'];sig=1/(1+np.exp(-a));derivatives.append(sig+a*sig*(1-sig));z=a*sig
    q=s['net.4.weight'].reshape(-1)*s['y_scale']
    for i,d in zip([2,0],reversed(derivatives)):q=s[f'net.{i}.weight'].T@(q*d)
    return q/s['x_scale']

def scalar_stats(a):
    a=sorted(a);n=len(a)
    if not n:return None
    def q(p):
        k=(n-1)*p;i=int(k);return a[i]+(k-i)*(a[min(i+1,n-1)]-a[i])
    return dict(mean=math.fsum(a)/n,median=statistics.median(a),p10=q(.1),p90=q(.9),min=a[0],max=a[-1])

def close(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:close(a[k],b[k])
    elif isinstance(a,(float,int)) and not isinstance(a,bool):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-14),(a,b)
    else:assert a==b,(a,b)

def main(out):
    setup();f=json.loads((out/'prediction_freeze.json').read_bytes());opening=json.loads((out/'reference_open.json').read_bytes());assert f['utc']<opening['first_source_clean_read_utc']
    s=json.loads((HERE/'selection.json').read_bytes());parent=json.loads(Path('research_log/T060A/cohort.json').read_bytes());assert sha('research_log/T060A/cohort.json')==s['cohort_sha256']
    expected_table=[dict(index=r['index'],bank_index=r['bank_index'],image_id=r['image_id'],active=r['gate']['active'],included=sum(r['gate']['active'])>0) for r in parent['anchors']]
    assert expected_table==s['all80'] and [r for r in parent['anchors'] if sum(r['gate']['active'])>0]==s['anchors']
    records=json.loads((out/'predictions.json').read_bytes());table=json.loads((out/'alignment_table.json').read_bytes());result=json.loads((out/'result.json').read_bytes());refs=torch.load(out/'reference_gradients.pt',weights_only=True,map_location='cpu');source=json.loads((out/'source_clean_reads.json').read_bytes())
    validate({r['path']:r['sha256'] for r in source});clean={r['image_id']:load_image(r['path']).numpy().astype(float) for r in source};states={h:torch.load(p,weights_only=True,map_location='cpu')['state_dict'] for h,p in HEADS.items()};checks=[];ind=[];fields={}
    assert len(records)==len(table)==len(s['anchors'])==f['rows']
    for r,t,c in zip(records,table,s['anchors']):
        assert all(r[k]==t[k]==c[k] for k in ['index','image_id','bank_index']);b=torch.load(out/r['file'],weights_only=True,map_location='cpu')
        assert all(thash(v)==r['tensor_hashes'][k] for k,v in b.items());assert torch.count_nonzero(b['gain'])==0
        assert thash(b['base'])==c['state_before']['y0'] and thash(b['grid'])==c['state_before']['grid'];gate=c['gate'];active=torch.tensor(gate['active']).float();signed=active*torch.where(torch.tensor(gate['winner'])==0,1.,-1.)
        expected=torch.cat((active,signed,torch.tensor(gate['evidence'],dtype=torch.float64))).float();assert torch.equal(b['x'][:12],expected) and torch.equal(b['x'][20:],b['grid'].flatten())
        assert torch.count_nonzero(b['J'][:12])==0 and torch.count_nonzero(b['J'][20:])==0 and torch.count_nonzero(b['J'][:,~active.bool()])==0
        reference=refs[r['bank_index']].numpy().astype(float);analytic=reference_analytic(b['low'].cuda(),b['raw'].cuda(),gate['active'],clean[r['image_id']]);err=float(np.max(abs(analytic-reference)));rel=float(np.linalg.norm(analytic-reference)/max(np.linalg.norm(reference),1e-30));assert err<=1e-9+3e-5*float(np.max(abs(reference))) and rel<=3e-5
        ref=reference.flatten();rn=math.sqrt(math.fsum(v*v for v in ref));row=dict(reference_norm=rn);check=dict(bank_index=r['bank_index'],analytic_max_abs_error=err,analytic_relative_error=rel)
        for h in HEADS:
            q=q_analytic(states[h],b['x'].numpy());np.testing.assert_allclose(q,b['q_'+h].numpy(),rtol=3e-5,atol=2e-6)
            g=b['g_'+h].double().numpy();np.testing.assert_allclose(b['J'].double().numpy().T@b['q_'+h].double().numpy(),g,rtol=3e-5,atol=2e-7)
            norm=math.sqrt(math.fsum(v*v for v in g));dot=math.fsum(a*z for a,z in zip(g,ref));row[h]=dict(norm=norm,dot=dot,cosine=dot/(norm*rn) if norm*rn else 0.,positive_dot=dot>0);close(row[h],t[h]);assert math.isclose(norm,r['norms'][h],rel_tol=1e-12)
            check['q_'+h+'_max_abs_error']=float(np.max(abs(q-b['q_'+h].numpy())))
        row['nondegenerate']=rn>1e-12 and min(row['E']['norm'],row['014']['norm'])>1e-12;close(rn,t['reference_norm']);assert row['nondegenerate']==t['nondegenerate'];ind.append(row);checks.append(check);fields[r['bank_index']]={k:v for k,v in b.items() if k not in ['low','base']}
    eligible=[r for r in ind if r['nondegenerate']];n=len(eligible);hs={}
    for h in HEADS:
        hs[h]=dict(positive_dot_fraction=sum(r[h]['positive_dot'] for r in eligible)/n if n else 0.,wrong_sign_count=sum(r[h]['dot']<0 for r in eligible),zero_dot_count=sum(r[h]['dot']==0 for r in eligible),cosine=scalar_stats([r[h]['cosine'] for r in eligible]),dot=scalar_stats([r[h]['dot'] for r in eligible]),norm_all=scalar_stats([r[h]['norm'] for r in ind]),norm_nondegenerate=scalar_stats([r[h]['norm'] for r in eligible]))
    close(hs,result['heads']);assert n==result['nondegenerate'];close(scalar_stats([r['reference_norm'] for r in ind]),result['reference_norm_all']);close(scalar_stats([r['reference_norm'] for r in eligible]),result['reference_norm_nondegenerate'])
    d=dict(positive_fraction_E_minus_014=hs['E']['positive_dot_fraction']-hs['014']['positive_dot_fraction'],median_cosine_E_minus_014=hs['E']['cosine']['median']-hs['014']['cosine']['median'] if n else None,wrong_sign_014_minus_E=hs['014']['wrong_sign_count']-hs['E']['wrong_sign_count']);close(d,result['deltas'])
    good=n>=40 and hs['E']['positive_dot_fraction']>=.9 and hs['E']['cosine']['median']>=.6 and d['positive_fraction_E_minus_014']>=0 and d['median_cosine_E_minus_014']>=0 and (d['median_cosine_E_minus_014']>=.05 or d['wrong_sign_014_minus_E']>=2)
    label='common-gain source direction audit lacks nondegenerate coverage' if n<40 else 'T059-E is a common-gain direction candidate for one later finite-step test' if good else 'T059-E does not improve the deployed common-gain direction enough to justify integration';assert label==result['classification']
    validate({str(out/n):h for n,h in f['files'].items()});validate(f['inputs']);validate(f['source_bindings']);assert f['head_state_before']==f['head_state_after']
    for h,state in states.items():assert {k:thash(v) for k,v in state.items()}==f['head_state_before'][h]
    save_tensor(out/'prediction_fields.pt',fields);atomic_json(out/'verification.json',dict(status='PASS',rows=len(records),checks=checks,independent='literal fixed-state gain derivative and region sums; explicit normalized SiLU MLP derivative; NumPy J-transpose-q; scalar summaries, exact selection/hash replay',classification=label,utc=utc()));print('INDEPENDENT_PASS',len(records),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
