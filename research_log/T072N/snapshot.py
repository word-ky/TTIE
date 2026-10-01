"""One host snapshot of GPU inventory and allowlisted process metadata only."""
import datetime
import json
import os
import subprocess

commands=[]
def query(args):
    p=subprocess.run(args,text=True,capture_output=True)
    record={'argv':args,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
    commands.append(record)
    return record

gpu=query(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'])
apps=query(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits'])
blockers=[]
for pid in (1337099,1337100):
    info=query(['ps','-p',str(pid),'-o','user:32=,pid=,ppid=,comm:40=,lstart=,etimes='])
    item={'pid':pid,'metadata':info,'parent_chain':[]}
    if not info['stdout'].strip():
        item['status']='DISAPPEARED' if info['returncode']==1 and not info['stderr'].strip() else 'METADATA_UNAVAILABLE'
    else:
        item['status']='RUNNING'
        try: item['cwd']={'status':'READABLE','path':os.readlink('/proc/'+str(pid)+'/cwd')}
        except OSError as e: item['cwd']={'status':'UNAVAILABLE','error_type':type(e).__name__,'errno':e.errno,'message':str(e)}
        parent=int(info['stdout'].split()[2])
        for level in range(1,5):
            if parent<=0:break
            row=query(['ps','-p',str(parent),'-o','pid=,ppid=,comm='])
            item['parent_chain'].append({'level':level,**row})
            if not row['stdout'].strip():break
            parent=int(row['stdout'].split()[1])
    blockers.append(item)
print(json.dumps({'task':'T072-N','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gpu_query':gpu,'process_query':apps,'blockers':blockers,'commands':commands,'accounting':{'process_interventions':0,'inference_runs':0,'input_payload_reads':0,'reference_reads':0,'metrics':0}},indent=2))
