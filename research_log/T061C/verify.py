"""Independent fixed lookup and aggregate replay (no analysis functions reused)."""
import argparse,csv,hashlib,json,statistics
from pathlib import Path
from decimal import Decimal,localcontext

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();o=a.out
    intent=json.loads((o/'intent.json').read_bytes());h=hashlib.sha256((o/'intent.json').read_bytes()).hexdigest()
    assert h==json.loads((o/'intent_hash.json').read_bytes())['sha256']
    assert intent['k_star']==11
    assert hashlib.sha256((a.root/'research_log/T061B/result/manifest.json').read_bytes()).hexdigest()==intent['source_manifest_sha256']
    tables=[]
    for name,record in intent['inputs'].items():
        data=(a.root/name).read_bytes();assert hashlib.sha256(data).hexdigest()==record['sha256']
        assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==record['git_blob']
        tables.append(list(csv.DictReader(data.decode().splitlines())))
    steps,images=tables;reported=json.loads((o/'paired.json').read_bytes());result=json.loads((o/'result.json').read_bytes())
    assert len(images)==len(reported)==100
    delta=[];tail=[];ssim=[]
    for i,(image,r) in enumerate(zip(images,reported)):
        assert int(image['index'])==r['index']==i and image['low']==r['low'] and r['step']==11
        states=[x for x in steps if x['method']=='common' and x['low']==image['low']]
        assert len(states)==41 and {int(x['step']) for x in states}==set(range(41))
        chosen=next(x for x in states if int(x['step'])==11)
        for metric in ['psnr','ssim']:
            assert float(chosen[metric])==r['step11_'+metric]
            for method,base in [('common','t036'),('baseline','t026')]:
                anchor=float(image[method+'_selected_'+metric]);assert anchor==r[base+'_'+metric]
                assert float(chosen[metric])-anchor==r['delta_'+metric+'_'+base]
        delta.append(float(chosen['psnr'])-float(image['common_selected_psnr']))
        tail.append(float(chosen['psnr'])-float(image['baseline_selected_psnr']))
        ssim.append(float(chosen['ssim'])-float(image['common_selected_ssim']))
    def average(xs):
        with localcontext() as c:
            c.prec=80
            return float(sum(map(Decimal.from_float,xs),Decimal(0))/Decimal(100))
    mean=average(delta);med=(sorted(delta)[49]+sorted(delta)[50])/2;sm=average(ssim)
    errors=[abs(mean-result['stats']['mean_psnr_vs_t036']),abs(med-result['stats']['median_psnr_vs_t036']),abs(sm-result['stats']['mean_ssim_vs_t036'])]
    assert max(errors)<1e-12
    counts=lambda xs:dict(improve=len([x for x in xs if x>0]),regress=len([x for x in xs if x<0]),tie=xs.count(0))
    assert counts(delta)==result['stats']['counts_vs_t036'] and counts(tail)==result['stats']['counts_vs_t026']
    assert min(tail)==result['stats']['worst_psnr_vs_t026']
    gates=dict(mean_psnr=mean>=.20,median_psnr=med>0,regressions=counts(tail)['regress']<=29,worst_psnr=min(tail)>=-5.614,mean_ssim=sm>=-.001)
    assert gates==result['gates']
    label='source-chosen fixed stopping is a transferable T036 selector improvement' if all(gates.values()) else 'a single source-chosen fixed stopping step does not transfer sufficiently'
    assert label==result['classification']
    assert intent['frozen_utc']<result['first_development_quality_read_utc']
    receipt=dict(status='PASS',images=100,fixed_lookups=100,max_aggregate_error=max(errors),gates=gates,classification=label,intent_sha256=h)
    with (o/'verification.json').open('x') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(receipt))

if __name__=='__main__':main()
