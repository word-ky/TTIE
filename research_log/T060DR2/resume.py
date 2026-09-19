"""Resume the observed ENOSPC interruption; retain 34 pairs plus complete anchor34 A."""
import argparse,ast
from research_log.T060DR2 import infer
from research_log.T060DR2.preflight import *

def main(out):
    recovery=json.loads((HERE/'storage_recovery.json').read_bytes())
    assert sha(out/'progress.json')==recovery['progress_sha256']
    rows=json.loads((out/'progress.json').read_bytes());assert len(rows)==34
    for r in rows:
        for name,m in r['methods'].items():validate({str(out/f"{r['order']:03d}"/name/n):h for n,h in m['files'].items()})
    validate({str(out/'034/A'/n):h for n,h in recovery['anchor34_A']['files'].items()})
    source=Path(infer.__file__).read_text();assert sha(infer.__file__)==recovery['original_infer_sha256']
    changes={
        'setup();start=utc();out.mkdir(parents=True,exist_ok=False)':"setup();start=recovery['original_run_started_utc'];out.mkdir(parents=True,exist_ok=True)",
        ';rows=[]':";rows=json.loads((out/'progress.json').read_bytes())",
        "for i,r in enumerate(s['anchors']):":"for i,r in enumerate(s['anchors']):\n        if i<len(rows):continue",
        "for name in ['A','B']:":"for name in ['A','B']:\n            if i==34 and name=='A':\n                methods[name]=recovery['anchor34_A'];continue",
    }
    for old,new in changes.items():assert source.count(old)==1;source=source.replace(old,new)
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='main')
    scope=dict(vars(infer));scope['recovery']=recovery
    atomic_json(out/'resume_started.json',dict(utc=utc(),recovery_sha256=sha(HERE/'storage_recovery.json'),retained_complete_pairs=34,retained_extra_A=34,first_recomputed='034/B',original_run_started_utc=recovery['original_run_started_utc']))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'ENOSPC_resume_literal_infer','exec'),scope)
    scope['main'](out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
