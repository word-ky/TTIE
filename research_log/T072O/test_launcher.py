import copy,hashlib,json,tempfile,unittest
from pathlib import Path
import launcher

class Mock:
    def __init__(self,qualifies=True,fail=None,bad_binding=False):self.events=[];self.qualifies=qualifies;self.fail=fail;self.bad_binding=bad_binding
    def snapshot(self):
        self.events.append('snapshot')
        return {'gpus':[{'index':1,'uuid':'mock','name':'NVIDIA RTX A6000','free_mib':40960 if self.qualifies else 3495}],'processes':[]}
    def bindings(self,spec,bindings):
        self.events.append('bindings')
        if self.bad_binding:raise ValueError('synthetic binding mismatch')
    def input(self,smoke):self.events.append('input')
    def run(self,method,device,spec,binding,out):
        self.events.append(method)
        if method==self.fail:raise RuntimeError('injected CUDA out of memory')
        return {'method':method,'shape':[2160,3840,3],'dtype':'float32','finite':True,'output_sha256':hashlib.sha256(method.encode()).hexdigest(),'file_sha256':hashlib.sha256(b'synthetic').hexdigest(),'runtime_s':1.,'peak_allocated_bytes':512,'peak_reserved_bytes':1024}

class Tests(unittest.TestCase):
    def run_case(self,mock):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            return launcher.launch(mock,Path(folder)/'run')
    def test_gate_before_all_access(self):
        m=Mock(False);r=self.run_case(m)
        self.assertEqual(r['classification'],'BLOCKED_GPU_GATE');self.assertEqual(m.events,['snapshot'])
    def test_binding_before_input(self):
        m=Mock(bad_binding=True);r=self.run_case(m)
        self.assertEqual(r['classification'],'BLOCKED_PREFLIGHT');self.assertEqual(m.events,['snapshot','bindings'])
    def test_success_order_and_telemetry(self):
        m=Mock();r=self.run_case(m)
        self.assertEqual(m.events,['snapshot','bindings','input','retinexformer','snr_aware'])
        self.assertEqual(r['classification'],'UHDLL_NATIVE_BASELINES_SMOKE_PASS')
        self.assertEqual([x['run_count'] for x in r['runs']],[1,1])
        self.assertTrue(all(x['status']=='FROZEN' for x in r['runs']))
    def test_failure_terminal_no_retry(self):
        for method in launcher.METHODS:
            m=Mock(fail=method);r=self.run_case(m)
            self.assertEqual(r['classification'],'BLOCKED_NATIVE4K_'+method.upper())
            self.assertIn('injected CUDA out of memory',r['traceback'])
            self.assertEqual(m.events.count(method),1)
            self.assertEqual(m.events[-1],method)
    def test_contract_mutations_before_any_access(self):
        original=json.loads((launcher.HERE/'spec.json').read_bytes())
        mutations=[(('bindings','retinexformer'),'0'*64),(('bindings','snr_aware'),'0'*64),(('gate','free_mib'),1),(('gate','process_max_mib'),99999),(('options','resize'),True),(('options','precision_override'),True),(('entrypoints','snr_aware'),'alternative')]
        mutations += [(('smoke','path'),'/tmp/'+name+'/1003_UHD_LL.JPG') for name in ('reference','GT','clean')]
        mutations += [(('smoke','sha256'),'0'*64)]
        for path,value in mutations:
            with self.subTest(path=path,value=value):
                d=copy.deepcopy(original);d[path[0]][path[1]]=value;m=Mock()
                with self.assertRaises(ValueError):launcher.launch(m,'unused',json.dumps(d).encode())
                self.assertEqual(m.events,[])
    def test_fixed_invocations(self):
        spec,b=launcher.load_contract()
        for method in launcher.METHODS:
            args=launcher.invocation(method,spec,b[method],Path('mock'))
            expected=['--low',spec['smoke']['path'],'--checkpoint',b[method]['checkpoint'],'--config',b[method]['config'],'--out',str(Path('mock')/'outputs')]
            if method=='snr_aware':expected+=['--source',b[method]['source']]
            self.assertEqual(args,expected)

if __name__=='__main__':unittest.main(verbosity=2)
