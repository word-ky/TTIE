"""Replay raw snapshot locally; never contacts the host."""
import csv,json,hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
raw=(HERE/'snapshot.json').read_bytes()
data=json.loads(raw)
assert data['task']=='T072-N'
assert data['accounting']==dict(process_interventions=0,inference_runs=0,input_payload_reads=0,reference_reads=0,metrics=0)
assert sum(c['argv'][0]=='nvidia-smi' for c in data['commands'])==2
assert data['gpu_query']==data['commands'][0] and data['process_query']==data['commands'][1]
assert data['gpu_query']['returncode']==data['process_query']['returncode']==0
gpus=list(csv.reader(data['gpu_query']['stdout'].splitlines(),skipinitialspace=True))
apps=list(csv.reader(data['process_query']['stdout'].splitlines(),skipinitialspace=True))
assert gpus and all(len(g)==7 for g in gpus) and all(len(a)==4 for a in apps)
inventory=[]
for gpu in gpus:
    processes=[a for a in apps if a[0]==gpu[1]]
    qualifies=gpu[2]=='NVIDIA RTX A6000' and int(gpu[5])>=40960 and all(int(a[3])<=1024 for a in processes)
    inventory.append(dict(index=int(gpu[0]),uuid=gpu[1],name=gpu[2],free_mib=int(gpu[5]),qualifies=qualifies,processes=processes))
assert [b['pid'] for b in data['blockers']]==[1337099,1337100]
blockers=[]
for b in data['blockers']:
    assert b['metadata'] in data['commands']
    assert b['metadata']['returncode']==0 and b['status']=='RUNNING'
    fields=b['metadata']['stdout'].split()
    user,pid,parent,comm=fields[:4]
    assert int(pid)==b['pid'] and int(fields[-1])>0
    chain=[]
    assert len(b['parent_chain'])==4
    for row in b['parent_chain']:
        command={k:row[k] for k in ('argv','returncode','stdout','stderr')}
        assert command in data['commands'] and command['returncode']==0
        rp,rpp,rc=row['stdout'].split(maxsplit=2)
        assert int(rp)==int(parent)
        chain.append(dict(pid=int(rp),ppid=int(rpp),comm=rc.strip()))
        parent=rpp
    assert int(parent)==0
    blockers.append(dict(user=user,pid=int(pid),ppid=int(fields[2]),comm=comm,start_host_local=' '.join(fields[4:-1]),elapsed_seconds=int(fields[-1]),parents=chain,cwd=b['cwd']))
receipt={'task':'T072-N','classification':'GPU_BLOCKER_PROVENANCE_CHARACTERIZED','snapshot_utc':data['utc'],'snapshot_count':1,'inventory':inventory,'additional_a6000':[g['index'] for g in inventory if g['name']=='NVIDIA RTX A6000' and g['uuid'] not in ('GPU-9de4332b-3b09-a3b4-5589-30229f0c14fe','GPU-9c468c54-b125-4903-1476-77c4d63270be')],'blockers':blockers,'gate':{'minimum_free_mib':40960,'maximum_unrelated_process_mib':1024},'accounting':data['accounting'],'verification':'PASS'}
(HERE/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt))
