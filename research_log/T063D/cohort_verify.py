"""Independent binary-digest selection reconstruction from immutable path provenance."""
import hashlib,heapq,json
from pathlib import Path


def verify(directory):
    p=json.loads((directory/'provenance.json').read_bytes());m=json.loads((directory/'manifest.json').read_bytes())
    assert hashlib.sha256((directory/'provenance.json').read_bytes()).hexdigest()==m['provenance_sha256']
    forbidden=set()
    for r in p['prior_ledger']:
        if r['ever_reference_open']:forbidden.add(r['normal'])
    forbidden.update(r['normal'] for r in p['previous_cohort'])
    for receipt in p['later_reads']:forbidden.update(receipt['normal_paths'])
    allpairs={(r['low'],r['normal']) for r in p['train_pairs']}
    expected=heapq.nsmallest(100,[(hashlib.sha256(('T063D:'+low).encode()).digest(),low,normal) for low,normal in allpairs if normal not in forbidden])
    assert len(allpairs)==689 and len(forbidden)==516 and m['remaining']==173
    assert [dict(low=l,normal=n,selection_hash=h.hex()) for h,l,n in expected]==m['selected']
    chosen={r['normal'] for r in m['selected']}
    assert len(chosen)==100 and not chosen&forbidden and not chosen&{r['normal'] for r in p['original_development']}
    return dict(status='PASS',train=689,forbidden=516,remaining=173,selected=100,previous_reference_overlap=0,development_overlap=0,manifest_sha256=hashlib.sha256((directory/'manifest.json').read_bytes()).hexdigest())


if __name__=='__main__':
    d=Path('research_log/T063D');r=verify(d)
    with (d/'cohort_verification.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps(r))
