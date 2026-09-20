"""Independent NumPy feature/probability, CPU metric and SciPy support diagnosis."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.special import expit
from research_log.T063A.common import sha,write,utc
from research_log.T066A.verify import check_render
from research_log.T066B.run import read,transfer_labels


def independent_geometry(q,b,labels,exclude=False):
    labels=np.asarray(labels,dtype=bool);rows=[]
    for i in range(len(q)//28):
        d=cdist(q[i*28:(i+1)*28],b,metric='euclidean')
        if exclude:d[:,i*28:(i+1)*28]=np.inf
        for x in d:
            safe=np.flatnonzero(labels);unsafe=np.flatnonzero(~labels);si=int(safe[np.argmin(x[safe])]);ui=int(unsafe[np.argmin(x[unsafe])])
            rows.append(dict(d_safe=float(x[si]),d_unsafe=float(x[ui]),margin=float(x[ui]-x[si]),safe_bank_row=si,unsafe_bank_row=ui))
    return rows


def independent_scores(y,p,mask):
    cm=dict(safe_pred_safe=0,safe_pred_unsafe=0,unsafe_pred_safe=0,unsafe_pred_unsafe=0)
    for a,b,m in zip(y,p,mask):
        if m:cm[('safe' if a else 'unsafe')+'_pred_'+('safe' if b>=.5 else 'unsafe')]+=1
    n=cm['unsafe_pred_safe']+cm['unsafe_pred_unsafe']
    return dict(states=sum(bool(x) for x in mask),confusion=cm,unsafe_recall=cm['unsafe_pred_unsafe']/n if n else None)


def independent_distribution(rows,mask):
    selected=[r for r,m in zip(rows,mask) if m];n=len(selected)
    out=dict(states=n,fraction_m_positive=sum(r['margin']>0 for r in selected)/n if n else None)
    for key in ['d_safe','d_unsafe','margin']:
        a=sorted(r[key] for r in selected)
        def quantile(q):
            pos=(n-1)*q;i=int(pos);return a[i]+(a[min(i+1,n-1)]-a[i])*(pos-i)
        out[key]=dict(zip(['min','q25','median','q75','max'],[quantile(q) for q in [0,.25,.5,.75,1]]),mean=sum(a)/n) if n else None
    return out


def main(out):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    get=lambda n:json.loads((out/n).read_bytes());cfg=get('config.json')
    for p,h in cfg['source_binding'].items():assert sha(p)==h,p
    freeze=get('diagnostic_freeze.json');opened=get('reference_open.json');result=get('result.json')
    assert freeze['reference_reads']==0 and freeze['frozen_utc']<opened['first_reference_quality_read_utc']
    assert sha(out/'diagnostic_freeze.json')==opened['freeze_sha256']
    for n,k in [('development_target_free.json','development_sha256'),('transfer_target_free.json','transfer_sha256')]:assert sha(out/n)==freeze[k]
    prior=Path('research_log/T066A/evidence')
    for n,k in [('model.json','model_sha256'),('normalization.json','normalization_sha256'),('transfer_freeze.json','prior_transfer_sha256')]:assert sha(prior/n)==freeze[k]
    dt,tt=get('development_target_free.json'),get('transfer_target_free.json');model=read('model.json');norm=read('normalization.json')
    dev=read('development_features.json')['rows'];transfer=read('transfer_freeze.json')['rows'];feature_errors=[];prob_error=0.;reconstructed=[]
    for istransfer,records,table in [(False,dev,dt),(True,transfer,tt)]:
        independent,error=check_render(istransfer,records);feature_errors.append(error)
        x=np.concatenate([r['features'] for r in independent]);np.testing.assert_allclose(x,[r['features'] for r in table],rtol=0,atol=2e-12)
        z=(x-np.asarray(model['mean']))/np.asarray(model['scale']);p=expit(np.clip(np.einsum('ij,j->i',z,model['coefficients'])+model['intercept'],-30,30));prob_error=max(prob_error,float(np.max(np.abs(p-[r['p_safe'] for r in table]))));assert prob_error<1e-7
        for j,row in enumerate(table):
            assert row['index']==j//28 and row['step']==j%28 and row['base_step']==records[j//28]['base_step'] and row['render_hash']==records[j//28]['hashes'][j%28]
            assert set(row)=={'index','step','base_step','features','p_safe','render_hash'}
        reconstructed.append(x)
    labels=transfer_labels(transfer,True);saved=get('transfer_labels.json');metric_error=0.
    for a,b in zip(labels,saved):
        assert a['index']==b['index'] and a['step']==b['step'] and a['safe']==b['safe'];assert b['reference_read_utc']>=opened['first_reference_quality_read_utc']
        for k in ['psnr','t026_psnr','quality_margin']:metric_error=max(metric_error,abs(a[k]-b[k]))
    assert metric_error<1e-10
    training=read('training_table.json');dy=np.asarray([r['safe'] for r in training],dtype=bool);ty=np.asarray([r['safe'] for r in labels],dtype=bool)
    for a,b in zip(dt,training):assert a['features']==b['features'] and a['index']==b['index'] and a['step']==b['step']
    # Geometry uses exact frozen table bytes; independent renders above establish their meaning.
    z=lambda rows:(np.asarray([r['features'] for r in rows])-np.asarray(norm['mean']))/np.asarray(norm['scale'])
    dz,tz=z(dt),z(tt);dg=independent_geometry(dz,dz,dy,True);tg=independent_geometry(tz,dz,dy);distance_error=0.
    for calc,saved_geo in [(dg,get('development_geometry.json')),(tg,get('transfer_geometry.json'))]:
        for j,(a,b) in enumerate(zip(calc,saved_geo)):
            assert b['index']==j//28 and b['step']==j%28
            for k in ['safe_bank_row','unsafe_bank_row']:assert a[k]==b[k]
            for k in ['d_safe','d_unsafe','margin']:distance_error=max(distance_error,abs(a[k]-b[k]))
    assert distance_error<1e-10
    p=[r['p_safe'] for r in tt];prefix=[r['step']<=r['base_step'] for r in tt];base=[r['step']==r['base_step'] for r in tt]
    summaries={name:independent_scores(ty,p,mask) for name,mask in [('all',[True]*2800),('prefix',prefix),('base',base)]};assert summaries==result['transfer']
    for rows,mask,key in [(tg,~ty,'transfer_unsafe_support'),(dg,~dy,'development_loio_unsafe_support')]:
        dist=independent_distribution(rows,mask);expected=result[key]
        assert dist['states']==expected['states'] and dist['fraction_m_positive']==expected['fraction_m_positive']
        for k in ['d_safe','d_unsafe','margin']:
            for stat,value in dist[k].items():assert abs(value-expected[k][stat])<1e-10
    frac=sum(g['margin']>0 for g,y in zip(tg,ty) if not y)/int((~ty).sum());recall=summaries['all']['unsafe_recall'];false=sum(b and not y and pr>=.5 for b,y,pr in zip(base,ty,p))
    verdict=('TRANSFER_SUPPORT_SHIFT' if frac>=.75 else 'BOUNDARY_MISMATCH_WITH_UNSAFE_SUPPORT') if recall<.5 else ('SELECTED_TAIL_SPECIFIC_FAILURE' if false else 'NO_DIAGNOSTIC_FAILURE')
    assert verdict==result['classification'] and false==result['false_safe_base_states']
    expected_indices=[i for i,(b,y) in enumerate(zip(base,ty)) if b and (not y or i//28 in [16,86])];assert len(expected_indices)==len(result['unsafe_base_rows'])
    for j,r in zip(expected_indices,result['unsafe_base_rows']):
        assert r['index']==j//28 and r['step']==j%28 and r['safe']==int(ty[j]) and r['p_safe']==p[j]
        assert abs(r['quality_margin']-labels[j]['quality_margin'])<1e-10
        for k in ['d_safe','d_unsafe','margin','safe_bank_row','unsafe_bank_row']:assert abs(r[k]-tg[j][k])<1e-10
    write(out/'verification.json',dict(status='PASS',classification=verdict,development_feature_max_error=feature_errors[0],transfer_feature_max_error=feature_errors[1],probability_max_error=prob_error,metric_max_error=metric_error,distance_max_error=distance_error,development_states=2800,transfer_states=2800,model_fits=0,verified_utc=utc()));print(json.dumps(get('verification.json')),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out.resolve())
