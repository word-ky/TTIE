"""Generate mock-only evidence and content manifest; never use RealBackend."""
import hashlib,json,tempfile
from pathlib import Path
from test_launcher import Mock
from launcher import launch
HERE=Path(__file__).resolve().parent
evidence={}
for name,mock in [('blocked',Mock(False)),('bad_binding',Mock(bad_binding=True)),('success',Mock()),('failure_retinexformer',Mock(fail='retinexformer')),('failure_snr_aware',Mock(fail='snr_aware'))]:
    with tempfile.TemporaryDirectory(dir=HERE) as tmp:
        receipt=launch(mock,Path(tmp)/'run')
        evidence[name]={'events':mock.events,'receipt':receipt}
(HERE/'mock_evidence.json').write_bytes((json.dumps(evidence,indent=2)+'\n').encode())
names=['spec.json','baseline_bindings.json','launcher.py','worker.py','test_launcher.py','verify.py','mock_evidence.json','prepare.py','seal.py']
files={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in names}
manifest={'files':files,'root_sha256':hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'accounting':dict(real_inference_runs=0,real_input_payload_reads=0,reference_reads=0,real_metrics=0,process_interventions=0)}
(HERE/'seal.json').write_bytes((json.dumps(manifest,indent=2)+'\n').encode())
print(json.dumps(manifest))
