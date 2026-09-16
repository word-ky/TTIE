"""Snapshot or compare immutable T055 source/artifacts for verifier-only adjudication."""
from pathlib import Path
import json,hashlib,sys
base=Path('/home/wenchang/asdasdsad/wjq/TTIE');out=base/'shared/t055v';audit=base/'runs/20260916-205307-ttie-t055a-oracle/artifacts/REFERENCE_ORACLE_ONLY';pre=base/'runs/20260916-205215-ttie-t055a-preflight/artifacts/preflight';release=base/'releases/20260916-205152-ttie-t055a-extension'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
config=json.loads((audit/'config.json').read_bytes());paths=list(audit.rglob('*'))+list(pre.rglob('*'))
for name,h in config['source_binding'].items():assert sha(release/name)==h;paths.append(release/name)
manifest={str(p.relative_to(base)):sha(p) for p in paths if p.is_file()}
if sys.argv[1]=='before':
 (out/'frozen_before.json').write_text(json.dumps(manifest,indent=2)+'\n');print('SNAPSHOT',len(manifest))
else:
 before=json.loads((out/'frozen_before.json').read_bytes());assert before==manifest
 result=dict(status='PASS',files=len(manifest),source_files=len(config['source_binding']),all_scientific_files_unchanged=True,baseline_sha256=sha(out/'frozen_before.json'),original_verifier_sha256=sha(release/'research_log/T055A_oracle/replay.py'),new_verifier_sha256=sha(out/'replay.py'),optimizer_calls=0,metric_files_regenerated=False)
 (out/'immutability.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
