import argparse,json,io,zipfile,hashlib,zlib,math
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from PIL import Image
from research_log.T063A.common import sha,thash,write,utc
from research_log.T066A.verify import independent_predict
from research_log.T067B.verify import independent_choices
from scripts.evaluate_t026a import independent_ssim
from research_log.T070A.manifest import environment
from research_log.T071A.run import ARCHIVE,MANIFEST,MANIFEST_SHA,FINAL_SOURCE,HERE

def main(out):
    torch.set_num_threads(1);get=lambda n:json.loads((out/n).read_bytes());cfg=get('config.json');freeze=get('output_freeze.json');cohort=get('pairs_manifest.json');opened=get('reference_open.json');reads=get('reference_reads.json');table=get('per_image_metrics.json');result=get('result.json')
    for p,h in cfg['source_binding'].items():assert sha(p)==h,p
    mf=json.loads(MANIFEST.read_bytes());assert sha(MANIFEST)==MANIFEST_SHA==freeze['final_manifest_sha256'] and mf['source_commit']==FINAL_SOURCE==cfg['final_source_commit'] and mf['environment']==environment()
    for p,h in mf['source_binding'].items():assert sha(p)==h,p
    for a in mf['assets'].values():assert sha(a['path'])==a['sha256']
    model=json.loads(Path(mf['assets']['probability_model']['path']).read_bytes())
    assert sha(out/'output_freeze.json')==opened['output_freeze_sha256']==reads['output_freeze_sha256']==result['output_freeze_sha256']
    assert sha(out/'pairs_manifest.json')==freeze['pairs_manifest_sha256']==cfg['pairs_manifest_sha256'] and sha(out/'config.json')==freeze['config_sha256']
    assert cohort['frozen_utc']<cfg['started_utc']<freeze['frozen_utc']<opened['first_reference_access_utc']
    assert len(cohort['rows'])==len(freeze['rows'])==len(table)==len(reads['rows'])==100
    assert cohort['reference_reads']==freeze['reference_reads']==freeze['reference_member_reads']==0 and freeze['data_reads']==[r['staged_low'] for r in cohort['rows']]
    provenance=json.loads((HERE/'provenance.json').read_bytes());assert cohort['provenance']==provenance and ARCHIVE.stat().st_size==provenance['archive_bytes'] and sha(ARCHIVE)==provenance['archive_sha256']
    expected_low=[f'LOL-v2/Real_captured/Test/Low/low{k:05d}.png' for k in range(690,790)];expected_normal=[f'LOL-v2/Real_captured/Test/Normal/normal{k:05d}.png' for k in range(690,790)]
    assert cohort['low_member_reads']==expected_low and cohort['reference_member_reads']==0
    ps=[];ss=[];steps=[];error=0.
    with zipfile.ZipFile(ARCHIVE) as z:
        actual=[i.filename for i in z.infolist() if not i.is_dir() and i.filename.startswith('LOL-v2/Real_captured/Test/')]
        assert len(actual)==200 and set(actual)==set(expected_low+expected_normal)
        for i,(pair,row,metric,read) in enumerate(zip(cohort['rows'],freeze['rows'],table,reads['rows'])):
            assert pair['index']==row['index']==metric['index']==read['index']==i and pair['image_number']==metric['image_number']==690+i
            assert pair['low']==row['low']==metric['low']==expected_low[i] and pair['normal']==row['normal']==metric['normal']==read['member']==expected_normal[i]
            li=z.getinfo(pair['low']);ri=z.getinfo(pair['normal'])
            assert li.CRC==pair['low_crc32'] and ri.CRC==pair['normal_crc32'] and li.file_size==pair['low_bytes'] and ri.file_size==pair['normal_bytes']
            lowbytes=z.read(pair['low']);assert hashlib.sha256(lowbytes).hexdigest()==pair['low_sha256']==row['low_sha256']==sha(pair['staged_low'])
            directory=out/'outputs'/f'{i:03d}';assert sha(directory/'output.pt')==row['output_file_sha256'] and sha(directory/'decision.json')==row['decision_file_sha256']
            decision=json.loads((directory/'decision.json').read_bytes());saved=torch.load(directory/'output.pt',map_location='cpu',weights_only=True)
            assert thash(saved['image'])==row['output_hash']==decision['output_hash'] and thash(saved['state'])==row['selected_state_hash']==decision['selected_state_hash']
            probs=independent_predict(decision['features'],model);np.testing.assert_allclose(probs,decision['probabilities'],rtol=0,atol=1e-12)
            k=dict(independent_choices(decision['values'],probs,decision['k_rho']))[.875]
            fs=min(j for j in range(decision['k_rho']+1) if probs[j]>=.5)
            assert k==decision['selected_step']==row['selected_step']==metric['selected_step'] and fs==decision['k_FS']==row['k_FS']
            assert decision['reference_reads']==row['reference_reads']==0 and decision['optimizer_runs']==1 and decision['optimizer_updates']==27 and decision['model_fits']==0
            assert opened['first_reference_access_utc']<=read['opened_utc'] and freeze['frozen_utc']<read['opened_utc']
            raw=z.read(pair['normal']);assert hashlib.sha256(raw).hexdigest()==read['sha256']==metric['normal_sha256'] and zlib.crc32(raw)==read['crc32']==pair['normal_crc32']
            with Image.open(io.BytesIO(raw)) as im:ref=torch.from_numpy(np.asarray(im.convert('RGB')).copy()).float().div(255).double()
            image=saved['image'].squeeze(0).permute(1,2,0).double();assert image.shape==ref.shape
            p=float(-10*torch.log10((image-ref).square().mean()));s=independent_ssim(image.numpy(),ref.numpy());error=max(error,abs(p-metric['psnr']),abs(s-metric['rgb_ssim']));ps.append(p);ss.append(s);steps.append(k)
    assert error<1e-11
    assert abs(math.fsum(ps)/100-result['mean_psnr'])<1e-11 and abs(math.fsum(ss)/100-result['mean_rgb_ssim'])<1e-11
    assert abs((sorted(ps)[49]+sorted(ps)[50])/2-result['median_psnr'])<1e-11
    assert result['selected_steps']==dict(min=min(steps),median=(sorted(steps)[49]+sorted(steps)[50])/2,max=max(steps),histogram={str(k):v for k,v in sorted(Counter(steps).items())})
    assert result['images']==100 and result['optimizer_runs']==freeze['optimizer_runs']==100 and result['optimizer_updates']==2700 and result['model_fits']==0 and result['inference_reference_reads']==0
    assert result['classification']=='OFFICIAL_LOLV2_REAL_TEST_RESULT_FROZEN' and sha(MANIFEST)==MANIFEST_SHA
    receipt=dict(status='PASS',classification=result['classification'],complete_official_pairs=100,independent_metric_max_abs_error=error,output_freeze_sha256=sha(out/'output_freeze.json'),reference_ordering_verified=True,independent_reference_reads=100,inference_reference_reads=0,optimizer_runs=0,model_fits=0,verified_utc=utc())
    write(out/'verification.json',receipt);print(json.dumps(receipt),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out.resolve())
