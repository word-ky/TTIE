import subprocess,sys
import pytest
from research_log.T060A.core import *
from research_log.T060A.verify import reference_analytic

def test_reference_adjoint():
    base=torch.linspace(.01,1,3*13*17).reshape(1,3,13,17);clean=base.flip(-1);m=Exposure(base,torch.zeros(1,2,2,2));assert torch.equal(m(),base);g,=torch.autograd.grad((m().double()-clean.double()).square().mean(),m.u);np.testing.assert_allclose(g.numpy(),reference_analytic(base.numpy(),clean.numpy()),rtol=3e-5,atol=1e-9)
def test_fixed_gates():
    assert 'insufficient' in classification(71,1,1)
    assert 'useful' in classification(72,.8,.4)
    assert 'does not transfer' in classification(80,.799,.9)
    assert 'does not transfer' in classification(80,1,.399)
def test_mutation(tmp_path):
    p=tmp_path/'bound';p.write_bytes(b'a');d={str(p):sha(p)};p.write_bytes(b'b')
    with pytest.raises(AssertionError):validate(d)
def test_firewall():
    program="from research_log.T060A.core import *;firewall([],ROOT/'unused-output');open(ROOT/'shared/t008/val2017/forbidden.jpg','rb')"
    r=subprocess.run([sys.executable,'-c',program],capture_output=True,text=True);assert r.returncode and 'target-free boundary' in r.stderr
