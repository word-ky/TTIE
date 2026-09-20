"""Independent immutable-source exclusion/selection replay, no selector import."""
import json,hashlib,subprocess,re,heapq
from pathlib import Path
from datetime import datetime,timezone
P=Path('research_log/T062CR1')
def sha(b):return hashlib.sha256(b).hexdigest()
def strings(value):
    if isinstance(value,str):yield value.replace(chr(92),'/')
    elif isinstance(value,list):
        for entry in value:yield from strings(entry)
    elif isinstance(value,dict):
        for key in ('path','normal','low','file','source','target'):
            if key in value:yield from strings(value[key])
a=json.loads((P/'allowlist.json').read_bytes());m=json.loads((P/'manifest.json').read_bytes())
assert sha((P/'allowlist.json').read_bytes())==m['allowlist_sha256']
assert sha((P/'ever_reference_open.json').read_bytes())==m['ledger_sha256']
assert sha((P/'discovery.json').read_bytes())==a['discovery_sha256']
assert sha((P/'branch_snapshot.json').read_bytes())==a['branch_snapshot_sha256']
assert sha((P/'path_provenance.json').read_bytes())==a['path_provenance_sha256']
used=set();reasons={};original=set();projections={};pairs=None
batch=subprocess.Popen(['git','cat-file','--batch'],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for s in a['sources']:
    batch.stdin.write((s['commit']+':'+s['path']+'\n').encode());batch.stdin.flush()
    header=batch.stdout.readline().decode().split();assert header[0]==s['blob']
    raw=batch.stdout.read(int(header[2]));batch.stdout.read(1);assert sha(raw)==s['sha256']
    normals=set()
    if s['kind']=='historical_ledger':
        d=json.loads(raw);pairs=[(r['low'],r['normal']) for r in d['candidates']]
        normals={r['normal'] for r in d['candidates'] if bool(r['excluded_by'])}
    elif s['kind']=='used_manifest':
        normals={r['normal'] for r in json.loads(raw)['selected']};original=normals.copy()
    elif s['kind']=='access_record':
        text=raw.decode('utf-8-sig');values=[]
        for key in s['fields']:
            if key=='root':values.append(json.loads(text));continue
            hits=list(re.finditer('"'+re.escape(key)+'"'+r'\s*:',text));assert hits,key
            # Discovery stores the last occurrence of an access key in a record.
            values.append(json.JSONDecoder().raw_decode(text[hits[-1].end():].lstrip())[0])
        for value in values:
            for path in strings(value):
                match=re.search(r'Train/Normal/[^/]+\.png$',path)
                if match:normals.add(match[0])
    if s['kind'] in ['historical_ledger','used_manifest','access_record']:
        projections[s['id']]=sorted(normals)
    for normal in normals:reasons.setdefault(normal,[]).append(s['id'])
    used.update(normals)
batch.stdin.close();assert batch.wait()==0
projected=json.loads((P/'path_provenance.json').read_bytes())
assert projections=={s['id']:s['normal_paths'] for s in projected['sources']}
assert pairs==[(r['low'],r['normal']) for r in projected['train_pairs']]
assert len(pairs)==689 and used<={normal for _,normal in pairs}
ranked=heapq.nsmallest(100,((hashlib.sha256(('T062C-R1:'+low).encode()).digest(),low,normal) for low,normal in pairs if normal not in used))
expected=[dict(low=low,normal=normal,selection_hash=h.hex()) for h,low,normal in ranked]
assert expected==m['selected'] and len(expected)==100 and m['remaining']==689-len(used)>=100
assert m['excluded']==len(used) and not {r['normal'] for r in expected}&(used|original)
ledger=[dict(low=lo,normal=no,ever_reference_open=no in used,provenance=reasons.get(no,[])) for lo,no in pairs]
assert ledger==json.loads((P/'ever_reference_open.json').read_bytes())
rebuilt=dict(m,selected=expected);assert sha((json.dumps(rebuilt,indent=2)+'\n').encode())==sha((P/'manifest.json').read_bytes())
r=dict(status='PASS',classification='COHORT_FREEZE_PASS',train=689,excluded=len(used),remaining=689-len(used),selected=100,ever_reference_overlap=0,T036_overlap=0,source_bindings=len(a['sources']),manifest_sha256=sha((P/'manifest.json').read_bytes()),verified_utc=datetime.now(timezone.utc).isoformat(),normal_image_opens=0,gpu_runs=0)
with (P/'verification.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(r))
