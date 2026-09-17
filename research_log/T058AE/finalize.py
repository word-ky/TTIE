"""Separate-process read-only new/old shard verification before final marker."""
import argparse,json,datetime,time,traceback
from pathlib import Path
from research_log.T058AE.storage import atomic_json,reopen,verify_shard0,coverage
from research_log.T058A_tangent.core import sha
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();start=time.perf_counter()
try:
    out=a.out;marker=json.loads((out/'new_shard_complete.json').read_bytes())
    assert marker['rows']==6322 and marker['range']==[1024,7345]
    assert marker['receipt_sha256']==sha(out/'receipt.json') and marker['manifest_sha256']==sha(out/'manifest.json')
    receipt=json.loads((out/'receipt.json').read_bytes())
    for n,h in receipt['source_bindings'].items():assert sha(n)==h
    new=reopen(out,6322,1024)
    old=verify_shard0(receipt['shard0_before']['run'],'research_log/T058AD_archives.json');assert old==receipt['shard0_before']==receipt['shard0_after']
    canonical=Path('research_log/T039A_result/stage_a/selection.json');assert sha(canonical)=='08227ee09f4cd428ee034d1d33cea7827037e853f2fe8b0a7efedcccfecc964c'
    old_rows=json.loads((Path(old['run'])/'artifacts/T058AD/manifest.json').read_bytes())['rows'];new_rows=json.loads((out/'manifest.json').read_bytes())['rows']
    combined=coverage(old_rows,new_rows,json.loads(canonical.read_bytes())['banks']);assert combined['rows']==7346 and combined['first']==0 and combined['last']==7345
    result=dict(classification='T058-A Stage A complete — all 7346 learned gradients frozen',separate_process=True,checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),new_shard=new,immutable_shard0=old,combined=combined,new_rows=6322,reused_rows=1024,receipt_sha256=sha(out/'receipt.json'),new_manifest_sha256=sha(out/'manifest.json'),seconds=time.perf_counter()-start,source_clean_opens=0,reference_gradient_executions=0,stage_b_executions=0,target_domain_access=0,official_test_access=0,optimizer_updates=0,persistent_scientific_state_changes=0)
    atomic_json(out/'combined_reopen.json',result)
    atomic_json(out/'complete.json',dict(classification=result['classification'],combined_rows=7346,new_range=[1024,7345],immutable_range=[0,1023],combined_reopen_sha256=sha(out/'combined_reopen.json'),receipt_sha256=sha(out/'receipt.json'),manifest_sha256=sha(out/'manifest.json'),completed_utc=result['checked_utc']))
    print(json.dumps(dict(classification=result['classification'],new_rows=6322,reused_rows=1024,combined_rows=7346,manifest_sha256=sha(out/'manifest.json'))),flush=True)
except Exception:
    (a.out/'finalize_error.txt').write_text(traceback.format_exc());raise
