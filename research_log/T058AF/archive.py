from pathlib import Path
import json,hashlib,tarfile,shutil
root=Path('/home/wenchang/asdasdsad/wjq/TTIE');run=root/'runs/20260917-153308-ttie-t058af-stageb';release=root/'releases/20260917-153229-ttie-t058af-stageb';out=root/'shared/t058af';backup=Path('/media/wenchang/F/wjq/TTIE/shared/t058af')
out.mkdir(parents=True,exist_ok=True);backup.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
binding=json.loads((release/'research_log/T058AF_source_binding.json').read_bytes())
for n,h in binding.items():assert sha(release/n)==h
# Archive a completed or failed sole run without changing its evidence.
path=out/'T058AF_evidence.tar.gz'
with tarfile.open(path,'w:gz') as t:
    for p in run.rglob('*'):
        if p.is_file():t.add(p,arcname=str(p.relative_to(run)))
shutil.copy2(path,backup/path.name);assert sha(path)==sha(backup/path.name)
r=dict(path=str(path),backup=str(backup/path.name),bytes=path.stat().st_size,sha256=sha(path),source208_unchanged=True,files={str(p.relative_to(run)):sha(p) for p in run.rglob('*') if p.is_file()})
(out/'T058AF_archives.json').write_text(json.dumps(r,indent=2)+'\n');shutil.copy2(out/'T058AF_archives.json',backup/'T058AF_archives.json');print(json.dumps({k:v for k,v in r.items() if k!='files'}))

# Compact local/Git export. Complete tensor archive stays in project home/F.
import csv,gzip,io
export=out/'export';export.mkdir(exist_ok=True)
stage=run/'artifacts/T058AF'
for name in ['receipt.json','complete.json','summary.json','reopen.json','stage_a_verified.json','source_opens.json']:
    if (stage/name).exists():shutil.copy2(stage/name,export/('T058AF_'+name))
shutil.copy2(out/'T058AF_archives.json',export/'T058AF_archives.json')
for name in ['run.sh','train.log']:shutil.copy2(run/name,export/('T058AF_'+name))
if (stage/'manifest.json').exists():
    manifest=json.loads((stage/'manifest.json').read_bytes())
    keys=[k for k in manifest['rows'][0] if k not in ['state_before','state_after','legacy_before','legacy_after']]
    with gzip.open(export/'T058AF_rows.csv.gz','wt',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
        for row in manifest['rows']:writer.writerow({k:row[k] for k in keys})
    compact={k:v for k,v in manifest.items() if k!='rows'}
    compact.update(raw_manifest_sha256=sha(stage/'manifest.json'),raw_manifest_path=str(stage/'manifest.json'),row_csv_gzip_sha256=sha(export/'T058AF_rows.csv.gz'),raw_archive_local_copy=False)
    (export/'T058AF_manifest_index.json').write_text(json.dumps(compact,indent=2)+'\n')
export_receipt={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in export.iterdir() if p.is_file() and p.name!='T058AF_export.json'}
(export/'T058AF_export.json').write_text(json.dumps(export_receipt,indent=2)+'\n');print(json.dumps(export_receipt))
