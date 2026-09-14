"""Freeze the requested path-hash cohort using accepted access receipts only."""
from pathlib import Path
import subprocess,json,hashlib,datetime
p=Path('research_log/T036A_cohort');p.mkdir(parents=True,exist_ok=True);bindings={}
def read(name):
    data=subprocess.check_output(['git','show','HEAD:'+name]);bindings[name]=hashlib.sha256(data).hexdigest();return json.loads(data)
ledger=read('research_log/T032A_cohort/ledger.json')
for name,h in ledger['provenance'].items():
    data=subprocess.check_output(['git','show','HEAD:'+name.replace('\\','/')])
    # The accepted T032 ledger recorded this one Windows CRLF working copy.
    if name.replace('\\','/')=='research_log/T030A_cohort/T023A_source_receipts.json':
        data=data.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')
    assert hashlib.sha256(data).hexdigest()==h,name
split=read('research_log/T022A_data/split.json');validation={r['normal'] for r in split['selected']}
c32=read('research_log/T032A_cohort/manifest.json');used32={r['normal'] for r in c32['selected']}
for name,expected in [('research_log/T032A_result/audit/evaluation_receipt.json',used32),
                      ('research_log/T033A_result/audit/evaluation_receipt.json',validation),
                      ('research_log/T034A_result/evidence/evaluation_receipt.json',validation),
                      ('research_log/T035A_result/evidence/evaluation_receipt.json',validation)]:
    receipt=read(name);paths=receipt.get('normal_opens',receipt.get('opened_normals'))
    if name=='research_log/T033A_result/audit/evaluation_receipt.json':paths=[r['path'] for r in paths]
    actual={x[x.index('Train/Normal/'):] for x in paths}
    assert actual==expected and len(paths)==100
candidates=[]
for original in ledger['candidates']:
    r={k:v for k,v in original.items() if k not in ['selection_hash','excluded_by']};reasons=list(original['excluded_by'])
    if r['normal'] in used32:reasons.append('T032-A actual accepted reference-open receipt')
    r.update(excluded_by=reasons,selection_hash=hashlib.sha256(r['low'].replace('\\','/').encode()).hexdigest());candidates.append(r)
eligible=sorted([r for r in candidates if not r['excluded_by']],key=lambda r:r['selection_hash']);assert len(eligible)>=100
stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
ex=dict(created_utc=stamp,base='35309c0d',prior_ledger=ledger,receipt_bindings=bindings,candidates=candidates,
        excluded_count=len(candidates)-len(eligible),eligible_count=len(eligible),
        audit='T032 accepted ledger accounts for earlier216 reference-used pairs. T032 actual reference receipt adds100. T033-T035 actual reference receipts equal originalvalidation100. Encoded SHA/IHDR metadata is not pixel decoding; no image payload read for selection. One legacy T023 source-receipt digest is explicitly checked in its recorded CRLF representation. Official test unused.')
(p/'exclusions.json').write_text(json.dumps(ex,indent=2)+'\n',encoding='utf-8')
manifest=dict(task='T036-A',created_utc=stamp,relative_root='LOL-v2/Real_captured/',
              selection='ascending SHA256(normalized relative low path), UTF-8, first100 after reference-use exclusions',
              eligible=len(eligible),exclusions_sha256=hashlib.sha256((p/'exclusions.json').read_bytes()).hexdigest(),selected=eligible[:100])
(p/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print('Eligible',len(eligible),'Excluded',ex['excluded_count'],'CohortSHA',hashlib.sha256((p/'manifest.json').read_bytes()).hexdigest())
