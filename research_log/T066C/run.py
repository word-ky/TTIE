import argparse,hashlib,json,os,time
from pathlib import Path
from datetime import datetime,timezone
from research_log.T066C.core import events,diagnose
HERE=Path('research_log/T066C')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def write(p,d):Path(p).write_text(json.dumps(d,indent=2),encoding='utf-8')
def load(p):return json.loads(Path(p).read_bytes())
def main(out):
    start=time.perf_counter();binding=load(HERE/'binding.json')
    for p,h in binding.items():assert sha(p)==h,p
    out.mkdir(parents=True,exist_ok=False);write(out/'config.json',dict(task='T066-C',source_commit=os.environ['TTIE_SOURCE_COMMIT'],bindings=binding,started_utc=utc(),device='CPU; fixed-sequence diagnostic only',optimizer_runs=0,model_fits=0))
    rows=load('research_log/T066B/evidence/transfer_target_free.json');table=events(rows);write(out/'events.json',table)
    write(out/'event_freeze.json',dict(event_sha256=sha(out/'events.json'),input_sha256=sha('research_log/T066B/evidence/transfer_target_free.json'),source_commit=os.environ['TTIE_SOURCE_COMMIT'],frozen_utc=utc(),label_reads=0))
    write(out/'label_join.json',dict(first_label_join_utc=utc(),freeze_sha256=sha(out/'event_freeze.json')))
    labels_binding=load(HERE/'evaluation_binding.json')
    for p,h in labels_binding.items():assert sha(p)==h,p
    labels=load('research_log/T066B/evidence/transfer_labels.json');result=diagnose(table,labels);write(out/'diagnosis.json',result)
    write(out/'receipt.json',dict(completed_utc=utc(),seconds=time.perf_counter()-start,classification=result['classification'],optimizer_runs=0,model_fits=0));print(json.dumps({k:v for k,v in result.items() if k!='unsafe_state_rows'}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out.resolve())
