"""Separate frozen diagnostic evidence verifier; no model or feature forward."""
import argparse,json
from pathlib import Path
import torch
from research_log.T058A_tangent.core import sha,thash,utc
from research_log.T059A.support import atomic_json
from research_log.T059D.core import bank_metrics
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();read=lambda n:json.loads((a.out/n).read_bytes());m=read('complete.json')
for n,h in m['files'].items():assert sha(a.out/n)==h,n
s=read('summary.json');r=read('receipt.json');banks=read('banks.json');rows=torch.load(a.out/'row_values.pt',map_location='cpu',weights_only=True)
assert r['row_tensor_hashes']=={k:thash(v) for k,v in rows.items()}
rebuilt,dp,dt,anchors=bank_metrics(rows['p'],rows['t'],rows['legacy_cosines'],rows['detail_cosines'],rows['legacy_mask'],rows['detail_mask'],rows['bank_indices'],rows['state_indices'],rows['global_indices'])
assert len(rebuilt)==80 and len(rows['p'])==1460 and torch.equal(dp,rows['delta_p']) and torch.equal(dt,rows['delta_t']) and torch.equal(anchors,rows['anchor_local_indices'])
for got,expected in zip(rebuilt,banks):assert all(got[k]==expected[k] for k in got)
relative=float(torch.nn.functional.huber_loss(dp,dt,delta=1.));absolute=float(torch.nn.functional.huber_loss(rows['p'],rows['t'],delta=1.))
assert relative==s['bank_relative_huber'] and absolute==s['unanchored_huber']
classification='C2 value failure is consistent with bankwise additive-offset miscalibration' if relative<=.07650849781930447 else 'C2 value failure is not explained by bankwise additive offsets'
assert classification==s['classification']==r['classification']==m['classification']
assert r['asset_hashes_before']==r['asset_hashes_after'] and r['input_tensor_hashes_before']==r['input_tensor_hashes_after'] and r['head_before']==r['head_after']
for n,h in r['asset_hashes_after'].items():assert sha(Path(n))==h,n
atomic_json(a.out/'verification.json',dict(classification=classification,all_hashes_reopened=True,rows=1460,banks=80,anchors=80,all_perbank_metrics_recomputed=True,unanchored_huber=absolute,bank_relative_huber=relative,classification_independently_recomputed=True,training_runs=0,completed_utc=utc()));print(json.dumps(dict(classification=classification,verification='PASS',relative=relative)),flush=True)
