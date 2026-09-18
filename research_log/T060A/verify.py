import argparse,math,statistics
from ttie.natural import load_image
from research_log.T060A.core import *
def reference_analytic(base,clean):
    base=np.asarray(base,dtype=np.float64);clean=np.asarray(clean,dtype=np.float64);h,w=base.shape[-2:]
    pixel=(2*(base-clean)/base.size*base*(2*math.log(2))).sum(axis=(0,1))
    yy=np.clip((np.arange(h)+.5)*8/h-.5,0,7);xx=np.clip((np.arange(w)+.5)*8/w-.5,0,7);yi=np.floor(yy).astype(int);xi=np.floor(xx).astype(int);yf=yy-yi;xf=xx-xi;g=np.zeros((8,8))
    for yind,yweight in [(yi,1-yf),(np.minimum(yi+1,7),yf)]:
        for xind,xweight in [(xi,1-xf),(np.minimum(xi+1,7),xf)]:np.add.at(g,(yind[:,None],xind[None,:]),pixel*yweight[:,None]*xweight[None,:])
    return g.reshape(1,1,8,8)
def scalar_stats(a):
    a=sorted(a);n=len(a)
    def q(p):
        k=(n-1)*p;i=int(k);return a[i]+(k-i)*(a[min(i+1,n-1)]-a[i])
    return dict(mean=math.fsum(a)/n,median=statistics.median(a),p10=q(.1),p90=q(.9),min=a[0],max=a[-1]) if n else None
def main(out):
    torch.set_num_threads(1);f=json.loads((out/'prediction_freeze.json').read_bytes());opening=json.loads((out/'reference_open.json').read_bytes());assert f['utc']<opening['first_source_clean_read_utc'];records=json.loads((out/'predictions.json').read_bytes());table=json.loads((out/'alignment_table.json').read_bytes());result=json.loads((out/'result.json').read_bytes());cohort=json.loads(Path('research_log/T060A/cohort.json').read_bytes());refs=torch.load(out/'reference_gradients.pt',weights_only=True,map_location='cpu');source=json.loads((out/'source_clean_reads.json').read_bytes());clean={r['image_id']:load_image(r['path']).numpy() for r in source};checks=[];ind=[];fields={}
    assert len(records)==len(table)==len(cohort['anchors'])==80
    for r,t,c in zip(records,table,cohort['anchors']):
        assert all(r[k]==t[k]==c[k] for k in ['index','image_id','bank_index']);b=torch.load(out/r['file'],weights_only=True,map_location='cpu');assert not torch.count_nonzero(b['u']);assert thash(b['base'])==c['state_before']['y0'] and thash(b['grid'])==c['state_before']['grid'];gate=c['gate'];active=torch.tensor(gate['active']).float();signed=active*torch.where(torch.tensor(gate['winner'])==0,1.,-1.);expected=torch.cat((active,signed,torch.tensor(gate['evidence'],dtype=torch.float64))).float();assert torch.equal(b['x'][:12],expected) and torch.equal(b['x'][20:],b['grid'].flatten())
        g=b['g'].numpy().astype(float).flatten();gR=refs[r['bank_index']].numpy().astype(float);np.testing.assert_allclose(b['J'].double().numpy().T@b['q'].double().numpy(),g,rtol=3e-5,atol=2e-7)
        analytic=reference_analytic(b['base'].numpy(),clean[r['image_id']]);err=float(np.max(abs(analytic-gR)));rel=float(np.linalg.norm(analytic-gR)/max(np.linalg.norm(gR),1e-30));assert err<=1e-9+3e-5*float(np.max(abs(gR))) and rel<=3e-5
        ref=gR.flatten();an=math.sqrt(math.fsum(v*v for v in g));bn=math.sqrt(math.fsum(v*v for v in ref));dot=math.fsum(a*b for a,b in zip(g,ref));cos=dot/(an*bn) if an*bn else 0.;row=dict(predicted_norm=an,reference_norm=bn,dot=dot,cosine=cos,positive_dot=dot>0,nondegenerate=an>1e-12 and bn>1e-12)
        for k in ['predicted_norm','reference_norm','dot','cosine']:assert math.isclose(row[k],t[k],rel_tol=1e-10,abs_tol=1e-14)
        assert row['positive_dot']==t['positive_dot'] and row['nondegenerate']==t['nondegenerate'];ind.append(row);checks.append(dict(bank_index=r['bank_index'],analytic_max_abs_error=err,analytic_relative_error=rel));fields[r['bank_index']]={k:b[k] for k in ['x','J','q','g','grid','raw','u']}
    eligible=[r for r in ind if r['nondegenerate']];n=len(eligible);positive=sum(r['positive_dot'] for r in eligible)/n if n else 0.;cos=scalar_stats([r['cosine'] for r in eligible]);assert n==result['nondegenerate'] and positive==result['positive_dot_fraction']
    expected={'cosine':cos,'dot':scalar_stats([r['dot'] for r in eligible])}
    for name,values in expected.items():
        for k,value in values.items():assert math.isclose(value,result[name][k],rel_tol=1e-10,abs_tol=1e-14)
    for name in ['predicted_norm','reference_norm']:
        for group,rows in [('all80',ind),('nondegenerate',eligible)]:
            for k,value in scalar_stats([r[name] for r in rows]).items():assert math.isclose(value,result['norms'][name][group][k],rel_tol=1e-10,abs_tol=1e-14)
    label='spatial-exposure projection is insufficiently active' if n<72 else 'frozen energy transfers useful spatial-exposure direction on source outer cohort' if positive>=.80 and cos['median']>=.40 else 'frozen energy does not transfer spatial-exposure direction under the fixed audit';assert label==result['classification'];validate({str(out/n):h for n,h in f['files'].items()});validate(f['inputs']);validate(f['source_bindings']);validate({r['path']:r['sha256'] for r in source});assert f['head_state_before']==f['head_state_after'] and f['optimizer_steps']==0
    save_tensor(out/'prediction_fields.pt',fields);atomic_json(out/'verification.json',dict(status='PASS',rows=80,checks=checks,independent='analytic RGB MSE/exposure derivative and bilinear adjoint, scalar norm/dot/cosine/quantiles/classification, NumPy J-transpose-q',classification=label,utc=utc()));print('INDEPENDENT_PASS80',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
