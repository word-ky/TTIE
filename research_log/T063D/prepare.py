"""Metadata-only extension of the accepted T062-C-R1 reference-use ledger."""
import hashlib, json, subprocess, re
from pathlib import Path
from datetime import datetime, timezone
HERE=Path('research_log/T063D')


def select(pairs,used):
    rows=[dict(low=r['low'].replace('\\','/'),normal=r['normal'].replace('\\','/')) for r in pairs if r['normal'].replace('\\','/') not in used]
    assert len(rows)>=100, 'Fewer than100 untouched pairs'
    for row in rows:row['selection_hash']=hashlib.sha256(('T063D:'+row['low']).encode()).hexdigest()
    return sorted(rows,key=lambda row:row['selection_hash'])[:100]


def main():
    source_bindings=[]
    tip=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    def read(path,commit=None):
        commit=commit or tip
        raw=subprocess.check_output(['git','show',commit+':'+path]);source_bindings.append(dict(commit=commit,path=path,sha256=hashlib.sha256(raw).hexdigest()))
        return json.loads(raw)
    prior=read('research_log/T062CR1/manifest.json');ledger=read('research_log/T062CR1/ever_reference_open.json');projections=read('research_log/T062CR1/path_provenance.json');allow=read('research_log/T062CR1/allowlist.json')
    batch=subprocess.Popen(['git','cat-file','--batch'],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    for s in allow['sources']:
        batch.stdin.write((s['commit']+':'+s['path']+'\n').encode());batch.stdin.flush();header=batch.stdout.readline().split();assert header[0].decode()==s['blob'];raw=batch.stdout.read(int(header[2]));batch.stdout.read(1)
        assert hashlib.sha256(raw).hexdigest()==s['sha256'];source_bindings.append(s)
    batch.stdin.close();assert batch.wait()==0
    pairs=projections['train_pairs'];lookup={r['low']:r['normal'] for r in pairs};used={r['normal'] for r in ledger if r['ever_reference_open']}
    assert used=={n for s in projections['sources'] for n in s['normal_paths']} and len(used)==416
    used.update(r['normal'] for r in prior['selected']);original=read('research_log/T036A_cohort/manifest.json')['selected'];assert {r['normal'] for r in original}<=used
    # All changed/new post-R1 branches were enumerated and inspected; no old snapshot branch changed.
    later=[]
    records=[('research_log/T062CR2/evidence/reference_reads.json','normal'),
             ('research_log/T063A/evidence/per_image.json','normal'),
             ('research_log/T063B/evidence/per_image.json','low'),
             ('research_log/T063C/evidence/per_image.json','low')]
    for path,key in records:
        rows=read(path);normals=sorted({r[key] if key=='normal' else lookup[r[key]] for r in rows})
        later.append(dict(source=path,normal_paths=normals));used.update(normals)
    assert len(pairs)==689 and len(used)==516
    rows=select(pairs,used);inventory=read('research_log/T036A_cohort/exclusions.json')['candidates'];metadata={r['low']:r for r in inventory}
    def write(name,obj):
        with (HERE/name).open('x',encoding='utf-8',newline='\n') as f:json.dump(obj,f,indent=2);f.write('\n')
    write('provenance.json',dict(prior_ledger=ledger,train_pairs=pairs,previous_cohort=prior['selected'],later_reads=later,original_development=original,source_bindings=source_bindings,prior_source_checks=len(allow['sources'])))
    write('manifest.json',dict(task='T063-D',created_utc=datetime.now(timezone.utc).isoformat(),selection='SHA256(T063D: + normalized relative low path), first100',train=689,excluded=len(used),remaining=689-len(used),selected=rows,normal_decodes=0,provenance_sha256=hashlib.sha256((HERE/'provenance.json').read_bytes()).hexdigest()))
    write('low_inputs.json',[dict(low=r['low'],low_sha256=metadata[r['low']]['low_sha256']) for r in rows])
    write('reference_inputs.json',[dict(normal=r['normal'],normal_sha256=metadata[r['low']]['normal_sha256']) for r in rows])
    print(json.dumps(dict(train=689,excluded=len(used),remaining=689-len(used),selected=100,manifest_sha256=hashlib.sha256((HERE/'manifest.json').read_bytes()).hexdigest())))


if __name__=='__main__':main()
