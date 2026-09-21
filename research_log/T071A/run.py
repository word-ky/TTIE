import argparse,json,os,time,zipfile,hashlib,zlib
from pathlib import Path
import torch
from research_log.T063A.common import sha,thash,write,utc
from research_log.T066A.run import ReadScope
from research_log.T070A.infer import FinalOurs,native_rgb
from research_log.T071A.core import pairs
HERE=Path('research_log/T071A')
ARCHIVE=Path('/media/wenchang/F/wjq/TTIE/shared/t022a/LOL-v2.zip')
MANIFEST=Path('research_log/T070A/evidence/FINAL_OURS_MANIFEST.json')
MANIFEST_SHA='e7f129d931d27531e5f6a14cd72e8c94734c474c40c3c403959cc8f5764e5ab9'
FINAL_SOURCE='aa4d920dff4b5b76751c24266e95ac9696d55d90'

def stage(out):
    provenance=json.loads((HERE/'provenance.json').read_bytes())
    assert ARCHIVE.stat().st_size==provenance['archive_bytes'] and sha(ARCHIVE)==provenance['archive_sha256']
    lowroot=out/'low';lowroot.mkdir()
    with zipfile.ZipFile(ARCHIVE) as z:
        rows=pairs(z.infolist());opened=[]
        for row in rows:
            data=z.read(row['low']);opened.append(row['low'])
            assert len(data)==row['low_bytes'] and zlib.crc32(data)==row['low_crc32']
            path=lowroot/Path(row['low']).name;path.write_bytes(data);row.update(low_sha256=hashlib.sha256(data).hexdigest(),staged_low=str(path))
    write(out/'pairs_manifest.json',dict(rows=rows,provenance=provenance,low_member_reads=opened,reference_member_reads=0,reference_reads=0,reference_hash_algorithm='CRC32 from central directory; SHA256 deferred until postfreeze',frozen_utc=utc()))
    return rows

def main(out):
    binding=json.loads((HERE/'binding.json').read_bytes())
    for path,h in binding.items():assert sha(path)==h,path
    assert sha(MANIFEST)==MANIFEST_SHA and json.loads(MANIFEST.read_bytes())['source_commit']==FINAL_SOURCE
    out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();rows=stage(out);model=FinalOurs(MANIFEST)
    torch.optim.Adam([torch.nn.Parameter(torch.zeros(1,device='cuda'))],lr=.03)
    cfg=dict(task='T071-A',source_commit=os.environ['TTIE_SOURCE_COMMIT'],final_source_commit=FINAL_SOURCE,final_manifest_sha256=MANIFEST_SHA,source_binding=binding,pairs_manifest_sha256=sha(out/'pairs_manifest.json'),started_utc=utc(),reference_reads=0,model_fits=0)
    write(out/'config.json',cfg);records=[];scope=ReadScope([Path(r['staged_low']) for r in rows],out/'outputs');(out/'outputs').mkdir()
    with scope:
        for row in rows:
            low=native_rgb(row['staged_low']);torch.cuda.synchronize();begin=time.perf_counter();result=model(low);torch.cuda.synchronize();seconds=time.perf_counter()-begin
            k=result['decision']['selected_step'];directory=out/'outputs'/f"{row['index']:03d}";directory.mkdir()
            torch.save(dict(image=result['image'],state=result['state']),directory/'output.pt')
            trace=result['trace'];decision=dict(**result['decision'],selected_state_hash=thash(result['state']),output_hash=thash(result['image']),trajectory_state_hash=thash(trace['states']),gradient_hash=thash(trace['gradients']),pre_box_hash=thash(trace['pre_box']),component_hash=thash(trace['components']),values=trace['values'],active=trace['active'].tolist(),winner=trace['winner'].tolist(),features=result['features'].tolist(),probabilities=result['probabilities'].tolist(),reference_reads=0,optimizer_runs=1,optimizer_updates=27,model_fits=0)
            write(directory/'decision.json',decision)
            records.append(dict(index=row['index'],image_number=row['image_number'],low=row['low'],normal=row['normal'],low_sha256=row['low_sha256'],selected_step=k,k_FS=decision['k_FS'],k_rho=decision['k_rho'],selected_state_hash=decision['selected_state_hash'],output_hash=decision['output_hash'],output_file_sha256=sha(directory/'output.pt'),decision_file_sha256=sha(directory/'decision.json'),seconds=seconds,reference_reads=0))
            print(json.dumps(dict(completed=len(records),seconds=seconds)),flush=True);del result,low
    assert len(records)==100 and scope.reads==[r['staged_low'] for r in rows]
    write(out/'output_freeze.json',dict(rows=records,pairs_manifest_sha256=sha(out/'pairs_manifest.json'),config_sha256=sha(out/'config.json'),final_manifest_sha256=sha(MANIFEST),reference_reads=0,reference_member_reads=0,data_reads=scope.reads,optimizer_runs=100,optimizer_updates=2700,model_fits=0,frozen_utc=utc(),inference_seconds=sum(r['seconds'] for r in records),stage_and_inference_seconds=time.perf_counter()-start))
    print('ALL_100_OFFICIAL_OUTPUTS_FROZEN',sha(out/'output_freeze.json'),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out.resolve())
