"""Inspect committed path-access fields only, across all preauthorization task branches."""
import subprocess,json,re,hashlib,concurrent.futures
from pathlib import Path
HERE=Path('research_log/T062CR1')
ACCESS=re.compile(r'"([^"\n]*(?:opens|reads|opened)[^"\n]*)"\s*:')
def git(*args):return subprocess.check_output(['git',*args])
def paths(v):
    if isinstance(v,str):yield v.replace(chr(92),'/')
    elif isinstance(v,list):
        for x in v:yield from paths(x)
    elif isinstance(v,dict):
        for k,x in v.items():
            if k in ['path','normal','low','file','source','target']:yield from paths(x)
def inspect(ref):
    entries=git('ls-tree','-r',ref['commit'],'research_log').decode().splitlines();rows=[];files=[]
    for e in entries:
        _,_,blob,name=e.split(None,3)
        if not re.match('research_log/T'+re.escape(ref['task'])+r'[/_.]',name) or not name.endswith('.json'):continue
        files.append(dict(path=name,blob=blob))
        raw=git('cat-file','blob',blob);s=raw.decode('utf-8-sig');access={}
        for m in ACCESS.finditer(s):
            key=m[1]
            if key in ['source_bindings','source_binding']:continue
            value,_=json.JSONDecoder().raw_decode(s[m.end():].lstrip())
            vals=list(paths(value));positive_count=value if isinstance(value,(int,float)) and value>0 else None
            # Persist only paths/access flags; never metric/outcome/condition fields.
            if vals or isinstance(value,(int,float,bool)):access[key]=dict(paths=vals,count=positive_count)
        if re.search(r'(reads|opens)\.json$',name):
            value=json.loads(s);access['root']=dict(paths=list(paths(value)),count=None)
        if access:
            normals=sorted({v[v.index('Train/Normal/'):] for a in access.values() for v in a['paths'] if 'Train/Normal/' in v})
            rows.append(dict(task=ref['task'],commit=ref['commit'],path=name,blob=blob,sha256=hashlib.sha256(raw).hexdigest(),fields=access,train_normals=normals))
    return dict(**ref,json_files=files),rows
if __name__=='__main__':
    snapshot=json.loads((HERE/'branch_snapshot.json').read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:results=list(pool.map(inspect,snapshot))
    report=dict(branches=[v[0] for v in results],access_sources=[r for _,rows in results for r in rows])
    (HERE/'discovery.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('branches',len(results),'JSON files',sum(len(r['json_files']) for r,_ in results),'access sources',len(report['access_sources']))
    for s in report['access_sources']:
        if s['train_normals']:print(s['task'],s['path'],len(s['train_normals']))
