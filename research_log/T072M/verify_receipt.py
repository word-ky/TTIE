"""Verify the observed prelaunch block from raw GPU/process CSV, no remote IO."""
import csv
import json
from pathlib import Path
import unittest
import copy

HERE = Path(__file__).resolve().parent
def verify(receipt, raw):
    assert receipt['task'] == 'T072-M'
    assert receipt['classification'] == 'BLOCKED_GPU_GATE'
    assert receipt['gate_checks'] == 1
    assert receipt['gate'] == {'device':'NVIDIA RTX A6000','minimum_free_mib':40960,'maximum_unrelated_process_mib':1024,'qualifying_devices':[]}
    assert receipt['smoke'] == {'name':'1003_UHD_LL.JPG','sha256':'cb89bcd019ac26a34665f07189483a0138e76e9b00714337c5e2919bae9062ca','geometry_wh':[3840,2160],'mode':'RGB','identity_source':'accepted T072-I declaration; payload unopened'}
    assert receipt['bindings'] == {'retinexformer':'a9a61665602618adf20c5fb6b05af22da5906c293876fe27342c05a5da833d00','snr_aware':'03ce8bb7f608051ec5c5fd92b7a3e3baea315a2d3978f4c62cc114d226cee875'}
    lines = raw.strip().splitlines()
    assert lines[0] == receipt['checked_utc']
    rows = list(csv.reader(lines[1:],skipinitialspace=True))
    assert all(len(r) in (7,4) for r in rows)
    gpus = [r for r in rows if len(r)==7]
    apps = [r for r in rows if len(r)==4]
    assert len(gpus)==2 and len({r[1] for r in gpus})==2
    qualifying = [int(g[0]) for g in gpus if g[2]=='NVIDIA RTX A6000' and int(g[5])>=40960 and all(int(a[3])<=1024 for a in apps if a[0]==g[1])]
    assert qualifying == []
    assert receipt['run_counts']=={'retinexformer':0,'snr_aware':0,'ours':0}
    assert receipt['statuses']=={'retinexformer':'UNRUN','snr_aware':'UNRUN','ours':'NOT_RUN'}
    for k in ('inference_runs','reference_reads','metrics','input_payload_reads','input_decodes','process_interventions'): assert receipt[k]==0
    assert receipt['outputs']==[]
    return {'classification':'BLOCKED_GPU_GATE','verification':'PASS','gate_checks':1,'free_mib':[int(g[5]) for g in gpus],'process_memory_mib':[int(a[3]) for a in apps],'inference_runs':0,'reference_reads':0,'metrics':0}

class ReceiptTests(unittest.TestCase):
    def test_reject_wrong_binding_or_accounting(self):
        original=json.loads((HERE/'receipt.json').read_text())
        raw=(HERE/'gate_raw.txt').read_text(encoding='utf-8-sig')
        for field in ('reference_reads','metrics','inference_runs','gate_checks'):
            r=copy.deepcopy(original); r[field]+=1
            with self.assertRaises(AssertionError): verify(r,raw)
        r=copy.deepcopy(original);r['bindings']['snr_aware']='0'*64
        with self.assertRaises(AssertionError):verify(r,raw)
    def test_reject_block_claim_for_qualifying_device(self):
        r=json.loads((HERE/'receipt.json').read_text())
        raw=(HERE/'gate_raw.txt').read_text(encoding='utf-8-sig')
        rows=list(csv.reader(raw.strip().splitlines()[1:],skipinitialspace=True))
        gpu=next(row for row in rows if len(row)==7)
        gpu[5]='45000'
        clean=r['checked_utc']+'\n'+', '.join(gpu)+'\n'+', '.join(next(row for row in rows if len(row)==7 and row[0]!=gpu[0]))
        with self.assertRaises(AssertionError):verify(r,clean)

if __name__=='__main__':
    result=verify(json.loads((HERE/'receipt.json').read_text()),(HERE/'gate_raw.txt').read_text(encoding='utf-8-sig'))
    (HERE/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))
    unittest.main(verbosity=2)
