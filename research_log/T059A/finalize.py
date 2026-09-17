import argparse,json
from pathlib import Path
from research_log.T059A.support import reopen_cache,atomic_json,sha,utc,verify_stage_a
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
r=json.loads((a.out/'receipt.json').read_bytes());assert r['rows']==7346 and sha(a.out/'manifest.json')==r['manifest_sha256']
for n,h in r['source_bindings'].items():assert sha(n)==h
banks=json.loads(Path('research_log/T039A_result/stage_a/selection.json').read_bytes())['banks']
_,_,stage=verify_stage_a(banks);assert {k:v for k,v in stage.items() if k!='verified_utc'}=={k:v for k,v in r['stage_a_before'].items() if k!='verified_utc'}
readback=reopen_cache(a.out,7346);readback['separate_process']=True;atomic_json(a.out/'independent_reopen.json',readback)
atomic_json(a.out/'complete.json',dict(classification='T059-A detail Jacobian cache valid',rows=7346,receipt_sha256=sha(a.out/'receipt.json'),manifest_sha256=sha(a.out/'manifest.json'),independent_reopen_sha256=sha(a.out/'independent_reopen.json'),completed_utc=utc()))
print('T059-A detail Jacobian cache valid',flush=True)
