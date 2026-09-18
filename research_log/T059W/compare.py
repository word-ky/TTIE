"""Cached degraded-image tensors are first decoded after the entire online run freezes."""
import argparse,hashlib
from research_log.T059W.core import *
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);out=p.parse_args().out;torch.set_num_threads(1)
f=json.loads((out/'online_freeze.json').read_bytes());assert f['rows']==16
for n,h in f['files'].items():assert sha(out/n)==h
stamp=utc();assert stamp>f['utc'];atomic_json(out/'comparison_open.json',dict(first_cached_tensor_read_utc=stamp,online_freeze_utc=f['utc']))
auth=json.loads(Path('research_log/T059W/authorization.json').read_bytes())
for n,h in auth['pinned_U_blobs'].items():
    path=U/Path(n).name;raw=path.read_bytes();assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==h,n
field=torch.load(U/'field.pt',weights_only=True,map_location='cpu');actions={r['bank_index']:r for r in json.loads((U/'actions.json').read_bytes())};uf=json.loads((U/'action_freeze.json').read_bytes());rows=[];cache_files={};records=json.loads((out/'online_records.json').read_bytes())
for r in records:
    a=actions[r['bank_index']];assert all(a[k]==r[k] for k in ['index','image_id','bank_index','state_index']);path=U/a['file'];assert sha(path)==uf['files'][a['file']];cache_files[str(path)]=sha(path)
    cache=torch.load(path,weights_only=True,map_location='cpu');online=torch.load(out/r['file'],weights_only=True,map_location='cpu');pos=a['position'];assert int(field['global_indices'][pos])==r['index'];assert torch.equal(online['y0'],cache['y0']) and torch.equal(online['mask'],cache['mask']),'STATE_INPUT_MISMATCH'
    target=dict(x=field['x'][pos],J=field['J'][pos],q=field['q'][pos],g=field['g'][pos],v1=cache['v1'],y1=cache['y1']);row={k:errors(online[k].numpy(),v.numpy()) for k,v in target.items()};row.update({k:r[k] for k in ['index','image_id','bank_index','state_index','selected_norm','selected_active_regions','active_regions']});row['passed']=passed(row);rows.append(row)
atomic_json(out/'comparison_table.json',rows)
for n,h in f['files'].items():assert sha(out/n)==h
for n,h in f['input_hashes_before'].items():assert sha(n)==h
for n,h in cache_files.items():assert sha(n)==h
result=dict(classification='active-path online target-free Jacobian bridge is reproducible across the fixed outer source images' if all(r['passed'] for r in rows) else 'active-path online target-free Jacobian bridge is not reproducible',passed=sum(r['passed'] for r in rows),total=16,failed_banks=[r['bank_index'] for r in rows if not r['passed']],online_freeze_utc=f['utc'],first_cached_tensor_read_utc=stamp,cached_comparison_inputs=cache_files,counters={k:0 for k in ['clean_reference_reads','reference_gradient_reads','target_domain_access','lolv2_access','official_test_access','inference_reference_leakage']},completed_utc=utc())
atomic_json(out/'result.json',result);print(json.dumps(result),flush=True)
