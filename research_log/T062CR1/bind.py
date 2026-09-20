"""Bind the reviewed access-source allow-list and emit path-only projections."""
import json,hashlib,subprocess
from pathlib import Path
P=Path('research_log/T062CR1');COMMIT='c7e02be34ccea01d7b473f08ac1b5a93e8de7457'
def git(*a):return subprocess.check_output(['git',*a])
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
    discovery=json.loads((P/'discovery.json').read_bytes());allow=[];projections=[]
    def bind(commit,name,kind,**kw):
        raw=git('show',commit+':'+name);blob=git('rev-parse',commit+':'+name).decode().strip()
        s=dict(id=str(len(allow)),commit=commit,path=name,blob=blob,sha256=sha(raw),kind=kind,**kw);allow.append(s);return s,raw
    a,raw=bind(COMMIT,'research_log/T036A_cohort/exclusions.json','historical_ledger');ledger=json.loads(raw)
    pairs=[{k:r[k] for k in ['low','normal']} for r in ledger['candidates']]
    projections.append(dict(id=a['id'],normal_paths=sorted(r['normal'] for r in ledger['candidates'] if r['excluded_by'])))
    a,raw=bind(COMMIT,'research_log/T036A_cohort/manifest.json','used_manifest');manifest=json.loads(raw)
    projections.append(dict(id=a['id'],normal_paths=sorted(r['normal'] for r in manifest['selected'])))
    prior_checks=[]
    for n,h in {**ledger['receipt_bindings'],**ledger['prior_ledger']['provenance']}.items():
        n=n.replace(chr(92),'/');a,raw=bind(COMMIT,n,'historical_binding')
        normalized=raw.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n') if n=='research_log/T030A_cohort/T023A_source_receipts.json' else raw
        assert sha(normalized)==h,n
        prior_checks.append(dict(path=n,accepted_digest=h,legacy_crlf=n=='research_log/T030A_cohort/T023A_source_receipts.json'))
    for source in discovery['access_sources']:
        a=dict(id=str(len(allow)),**{k:source[k] for k in ['task','commit','path','blob','sha256']},kind='access_record',fields=list(source['fields']))
        allow.append(a);projections.append(dict(id=a['id'],normal_paths=source['train_normals']))
    # Reviewed cases without a dedicated path-read receipt, plus latest T062 access mapping.
    extra={
      '055V':['research_log/T055V/replay.py','research_log/T055V_result/replay_run/run.sh','research_log/T022A_data/split.json'],
      '058AB':['research_log/T058AB/run.py'],
      '058AC':['research_log/T058AC/run.py'],
      '059BR':['research_log/T059BR/run.py'],
      '059C':['research_log/T059C/audit_split.py'],
      '061C':['research_log/T061C/analyze.py'],
      '062A':['research_log/T062A/infer.py','research_log/T062A/evaluate.py','research_log/T062A/evidence/reference_open.json'],
      '062B':['research_log/T062B/analyze.py','research_log/T062B/evidence/inputs.json']}
    for task,names in extra.items():
        commit=next(b['commit'] for b in discovery['branches'] if b['task']==task)
        for name in names:bind(commit,name,'reviewed_access_mapping',task=task)
    path_data=dict(train_pairs=pairs,sources=projections)
    (P/'path_provenance.json').write_text(json.dumps(path_data,indent=2)+'\n',encoding='utf-8')
    result=dict(authorization='ffe89df73bce4d30773b9401af3d92974a453cb3',sources=allow,path_provenance_sha256=sha((P/'path_provenance.json').read_bytes()),discovery_sha256=sha((P/'discovery.json').read_bytes()),branch_snapshot_sha256=sha((P/'branch_snapshot.json').read_bytes()),historical_bindings_verified=prior_checks)
    (P/'allowlist.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    union=set(n for s in projections for n in s['normal_paths'])
    print('bound',len(allow),'train',len(pairs),'excluded',len(union),'remaining',len(pairs)-len(union))
if __name__=='__main__':main()
