from pathlib import Path
import sys,json,datetime
release=Path('/home/wenchang/asdasdsad/wjq/TTIE/releases/20260917-131701-ttie-t058ad-shard0')
sys.path.insert(0,str(release))
from research_log.T058AD.storage import reopen,atomic_json
from research_log.T058A_tangent.core import sha
out=Path('/home/wenchang/asdasdsad/wjq/TTIE/runs/20260917-131739-ttie-t058ad-shard0/artifacts/T058AD')
marker=json.loads((out/'complete.json').read_bytes());assert marker['rows']==1024
assert sha(out/'receipt.json')==marker['receipt_sha256'] and sha(out/'manifest.json')==marker['manifest_sha256']
result=reopen(out,1024);result['separate_process']=True;result['checked_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
assert result['manifest_sha256']==marker['manifest_sha256']
atomic_json(out/'external_reopen.json',result);print(json.dumps(result))
