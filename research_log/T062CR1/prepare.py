"""One path-only unused-cohort selection from immutable provenance projections."""
import json,hashlib,subprocess
from pathlib import Path
from datetime import datetime,timezone
HERE=Path('research_log/T062CR1')
def sha(b):return hashlib.sha256(b).hexdigest()
def select(pairs,used):
    eligible=[dict(low=r['low'].replace(chr(92),'/'),normal=r['normal'].replace(chr(92),'/')) for r in pairs if r['normal'].replace(chr(92),'/') not in used]
    assert len(eligible)>=100
    for r in eligible:r['selection_hash']=sha(('T062C-R1:'+r['low']).encode())
    return sorted(eligible,key=lambda r:r['selection_hash'])[:100]
def main():
    allow=json.loads((HERE/'allowlist.json').read_bytes());projections=json.loads((HERE/'path_provenance.json').read_bytes())
    assert sha((HERE/'path_provenance.json').read_bytes())==allow['path_provenance_sha256']
    batch=subprocess.Popen(['git','cat-file','--batch'],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    for s in allow['sources']:
        batch.stdin.write((s['commit']+':'+s['path']+'\n').encode());batch.stdin.flush()
        header=batch.stdout.readline().decode().split();assert header[0]==s['blob']
        raw=batch.stdout.read(int(header[2]));batch.stdout.read(1)
        assert sha(raw)==s['sha256']
    batch.stdin.close();assert batch.wait()==0
    pairs=projections['train_pairs'];assert len(pairs)==689
    reasons={r['normal']:[] for r in pairs}
    for s in projections['sources']:
        for n in s['normal_paths']:assert n in reasons;reasons[n].append(s['id'])
    used={n for n,v in reasons.items() if v};rows=select(pairs,used)
    assert len(rows)==100 and not {r['normal'] for r in rows}&used
    ledger=[dict(low=r['low'],normal=r['normal'],ever_reference_open=bool(reasons[r['normal']]),provenance=reasons[r['normal']]) for r in pairs]
    utc=datetime.now(timezone.utc).isoformat()
    def write(n,d):
        with (HERE/n).open('x',encoding='utf-8',newline='\n') as f:json.dump(d,f,indent=2);f.write('\n')
    write('ever_reference_open.json',ledger)
    write('manifest.json',dict(task='T062-C-R1',created_utc=utc,relative_root='LOL-v2/Real_captured/',selection='SHA256(T062C-R1: + normalized relative low path), first100 ascending after complete historical reference-use union',train=689,excluded=len(used),remaining=689-len(used),selected=rows,allowlist_sha256=sha((HERE/'allowlist.json').read_bytes()),ledger_sha256=sha((HERE/'ever_reference_open.json').read_bytes()),normal_image_opens=0))
    print(json.dumps(dict(train=689,excluded=len(used),remaining=689-len(used),selected=100,manifest_sha256=sha((HERE/'manifest.json').read_bytes()))))
if __name__=='__main__':main()
